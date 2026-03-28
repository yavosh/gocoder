package config_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/yavosh/gocoder/internal/config"
)

func TestParseSourcesYAML(t *testing.T) {
	dir := t.TempDir()
	yamlContent := `repos:
  - url: https://github.com/golang/go
    path: src/
    weight: 1.5
  - url: https://github.com/cockroachdb/cockroach
articles:
  - url: https://go.dev/blog/
    type: blog_index
  - path: data/bookmarks.txt
    type: url_list
`
	path := filepath.Join(dir, "sources.yaml")
	if err := os.WriteFile(path, []byte(yamlContent), 0o644); err != nil {
		t.Fatal(err)
	}

	cfg, err := config.ParseSources(path)
	if err != nil {
		t.Fatalf("ParseSources: %v", err)
	}

	if len(cfg.Repos) != 2 {
		t.Fatalf("expected 2 repos, got %d", len(cfg.Repos))
	}
	if cfg.Repos[0].Path != "src/" {
		t.Errorf("expected path 'src/', got %q", cfg.Repos[0].Path)
	}
	if cfg.Repos[0].Weight != 1.5 {
		t.Errorf("expected weight 1.5, got %f", cfg.Repos[0].Weight)
	}
	if cfg.Repos[1].Weight != 0 {
		t.Errorf("expected default weight 0, got %f", cfg.Repos[1].Weight)
	}
	if len(cfg.Articles) != 2 {
		t.Fatalf("expected 2 articles, got %d", len(cfg.Articles))
	}
	if cfg.Articles[0].Type != "blog_index" {
		t.Errorf("expected type 'blog_index', got %q", cfg.Articles[0].Type)
	}
	if cfg.Articles[1].Path != "data/bookmarks.txt" {
		t.Errorf("expected path 'data/bookmarks.txt', got %q", cfg.Articles[1].Path)
	}
}

func TestParseSourcesLocalDir(t *testing.T) {
	dir := t.TempDir()
	yamlContent := `repos:
  - url: https://github.com/golang/go
  - dir: /home/user/projects/myrepo
    weight: 2.0
`
	path := filepath.Join(dir, "sources.yaml")
	if err := os.WriteFile(path, []byte(yamlContent), 0o644); err != nil {
		t.Fatal(err)
	}

	cfg, err := config.ParseSources(path)
	if err != nil {
		t.Fatalf("ParseSources: %v", err)
	}

	if len(cfg.Repos) != 2 {
		t.Fatalf("expected 2 repos, got %d", len(cfg.Repos))
	}
	if cfg.Repos[0].IsLocal() {
		t.Error("expected remote repo to not be local")
	}
	if !cfg.Repos[1].IsLocal() {
		t.Error("expected local repo to be local")
	}
	if cfg.Repos[1].Dir != "/home/user/projects/myrepo" {
		t.Errorf("expected dir '/home/user/projects/myrepo', got %q", cfg.Repos[1].Dir)
	}
}

func TestParseSourcesDefaultWeight(t *testing.T) {
	dir := t.TempDir()
	yamlContent := `repos:
  - url: https://github.com/golang/go
`
	path := filepath.Join(dir, "sources.yaml")
	if err := os.WriteFile(path, []byte(yamlContent), 0o644); err != nil {
		t.Fatal(err)
	}

	cfg, err := config.ParseSources(path)
	if err != nil {
		t.Fatalf("ParseSources: %v", err)
	}

	if cfg.Repos[0].EffectiveWeight() != 1.0 {
		t.Errorf("expected effective weight 1.0, got %f", cfg.Repos[0].EffectiveWeight())
	}
}
