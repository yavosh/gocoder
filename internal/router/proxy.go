package router

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httputil"
	"net/url"
	"time"
)

func NewProxy(cfg *Config) http.Handler {
	upstream, _ := url.Parse(cfg.Upstream)
	metrics := NewMetrics()

	proxy := &httputil.ReverseProxy{
		Director: func(req *http.Request) {
			req.URL.Scheme = upstream.Scheme
			req.URL.Host = upstream.Host
			req.Host = upstream.Host
		},
	}

	mux := http.NewServeMux()

	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"status":"ok"}`))
	})

	mux.Handle("/metrics", metrics.Handler())

	mux.HandleFunc("/v1/", func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		requestPath := r.URL.Path

		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "failed to read body", http.StatusBadRequest)
			return
		}

		resolvedModel := ""
		var reqBody map[string]any
		if err := json.Unmarshal(body, &reqBody); err == nil {
			if model, ok := reqBody["model"].(string); ok {
				resolvedModel = cfg.ResolveModel(model, requestPath)
				reqBody["model"] = resolvedModel
				rewritten, _ := json.Marshal(reqBody)
				body = rewritten
			}
		}

		r.Body = io.NopCloser(bytes.NewReader(body))
		r.ContentLength = int64(len(body))

		proxy.ServeHTTP(w, r)
		metrics.Record(resolvedModel, time.Since(start), false)
	})

	return mux
}
