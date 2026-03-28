package collector_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/yavosh/gocoder/internal/collector"
	"github.com/yavosh/gocoder/internal/config"
)

func TestDiscoverBlogURLs(t *testing.T) {
	// Mock blog index page with links
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		w.Write([]byte(`<html><body>
			<a href="/blog/error-handling">Error Handling</a>
			<a href="/blog/generics">Generics</a>
			<a href="/blog/modules">Modules</a>
			<a href="https://external.com/other">External</a>
		</body></html>`))
	}))
	defer srv.Close()

	urls, err := collector.DiscoverBlogURLs(context.Background(), srv.URL)
	if err != nil {
		t.Fatalf("DiscoverBlogURLs: %v", err)
	}

	// Should find the 3 relative blog links, resolved against the server URL
	if len(urls) < 3 {
		t.Errorf("expected at least 3 blog URLs, got %d: %v", len(urls), urls)
	}
}

func TestFetchMarkdownRepo(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test")
	}

	src := config.ArticleSource{
		URL:  "https://github.com/uber-go/guide",
		Type: "markdown_repo",
	}

	contents, err := collector.FetchMarkdownRepo(context.Background(), src)
	if err != nil {
		t.Fatalf("FetchMarkdownRepo: %v", err)
	}
	if len(contents) == 0 {
		t.Error("expected at least one markdown file")
	}
}

func TestFetchSitemapURLs(t *testing.T) {
	// Mock sitemap XML
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/sitemap.xml" {
			w.Header().Set("Content-Type", "application/xml")
			w.Write([]byte(`<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>http://example.com/page1</loc></url>
  <url><loc>http://example.com/page2</loc></url>
</urlset>`))
		}
	}))
	defer srv.Close()

	urls, err := collector.FetchSitemapURLs(context.Background(), srv.URL+"/sitemap.xml")
	if err != nil {
		t.Fatalf("FetchSitemapURLs: %v", err)
	}
	if len(urls) != 2 {
		t.Errorf("expected 2 URLs from sitemap, got %d", len(urls))
	}
}
