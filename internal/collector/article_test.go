package collector_test

import (
	"context"
	"testing"

	"github.com/yavosh/gocoder/internal/collector"
	"github.com/yavosh/gocoder/internal/config"
)

func TestFetchArticle(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test")
	}

	src := config.ArticleSource{
		URL:  "https://go.dev/doc/effective_go",
		Type: "single_page",
	}

	content, err := collector.FetchArticle(context.Background(), src)
	if err != nil {
		t.Fatalf("FetchArticle: %v", err)
	}
	if len(content) == 0 {
		t.Error("expected non-empty content")
	}
}
