package router_test

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/yavosh/gocoder/internal/router"
)

func TestRouteByModel(t *testing.T) {
	cfg := &router.Config{
		Models: map[string]router.ModelConfig{
			"codegen": {Model: "go-nemotron"},
			"fast":    {Model: "qwen3:30b-a3b"},
		},
		Routes: []router.Route{
			{Match: router.Match{Model: "go-*"}, Target: "codegen"},
			{Match: router.Match{Default: true}, Target: "codegen"},
		},
	}

	resolved := cfg.ResolveModel("go-nemotron", "/v1/chat/completions")
	if resolved != "go-nemotron" {
		t.Errorf("expected go-nemotron, got %s", resolved)
	}

	resolved = cfg.ResolveModel("unknown", "/v1/chat/completions")
	if resolved != "go-nemotron" {
		t.Errorf("expected default go-nemotron, got %s", resolved)
	}
}

func TestRouteByPath(t *testing.T) {
	cfg := &router.Config{
		Models: map[string]router.ModelConfig{
			"codegen":      {Model: "go-nemotron"},
			"autocomplete": {Model: "qwen3:30b-a3b"},
		},
		Routes: []router.Route{
			{Match: router.Match{Path: "/v1/completions"}, Target: "autocomplete"},
			{Match: router.Match{Default: true}, Target: "codegen"},
		},
	}

	resolved := cfg.ResolveModel("anything", "/v1/completions")
	if resolved != "qwen3:30b-a3b" {
		t.Errorf("expected qwen3:30b-a3b for /v1/completions, got %s", resolved)
	}

	resolved = cfg.ResolveModel("anything", "/v1/chat/completions")
	if resolved != "go-nemotron" {
		t.Errorf("expected go-nemotron for chat, got %s", resolved)
	}
}

func TestProxyForwardsRequest(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		if !strings.Contains(string(body), "go-nemotron") {
			t.Errorf("expected model go-nemotron in request body, got %s", body)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"choices":[{"message":{"content":"hello"}}]}`))
	}))
	defer backend.Close()

	cfg := &router.Config{
		Upstream: backend.URL,
		Models: map[string]router.ModelConfig{
			"codegen": {Model: "go-nemotron"},
		},
		Routes: []router.Route{
			{Match: router.Match{Default: true}, Target: "codegen"},
		},
	}

	proxy := router.NewProxy(cfg)
	srv := httptest.NewServer(proxy)
	defer srv.Close()

	reqBody := `{"model":"test","messages":[{"role":"user","content":"hi"}]}`
	resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(reqBody))
	if err != nil {
		t.Fatalf("proxy request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("expected 200, got %d", resp.StatusCode)
	}
}

func TestHealthEndpoint(t *testing.T) {
	cfg := &router.Config{
		Upstream: "http://localhost:11434",
		Models:   map[string]router.ModelConfig{},
		Routes:   []router.Route{},
	}

	proxy := router.NewProxy(cfg)
	srv := httptest.NewServer(proxy)
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/health")
	if err != nil {
		t.Fatalf("health request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("expected 200, got %d", resp.StatusCode)
	}
}
