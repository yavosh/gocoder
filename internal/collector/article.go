package collector

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/yavosh/gocoder/internal/config"
)

func FetchArticle(ctx context.Context, src config.ArticleSource) (string, error) {
	switch src.Type {
	case "single_page":
		return fetchURL(ctx, src.URL)
	case "url_list":
		return "", fmt.Errorf("url_list type requires FetchArticleList, not FetchArticle")
	case "blog_index", "sitemap", "markdown_repo":
		return "", fmt.Errorf("%s type returns multiple results; use FetchAllArticles", src.Type)
	default:
		return "", fmt.Errorf("unsupported article type: %s", src.Type)
	}
}

// FetchAllArticles handles all article source types, returning one result per page/file.
func FetchAllArticles(ctx context.Context, src config.ArticleSource) ([]string, error) {
	switch src.Type {
	case "single_page":
		content, err := fetchURL(ctx, src.URL)
		if err != nil {
			return nil, err
		}
		return []string{content}, nil
	case "url_list":
		return FetchArticleList(ctx, src)
	case "blog_index":
		urls, err := DiscoverBlogURLs(ctx, src.URL)
		if err != nil {
			return nil, err
		}
		var results []string
		for _, u := range urls {
			content, err := fetchURL(ctx, u)
			if err != nil {
				continue // skip failed fetches
			}
			results = append(results, content)
		}
		return results, nil
	case "markdown_repo":
		return FetchMarkdownRepo(ctx, src)
	case "sitemap":
		urls, err := FetchSitemapURLs(ctx, src.URL+"/sitemap.xml")
		if err != nil {
			return nil, err
		}
		var results []string
		for _, u := range urls {
			content, err := fetchURL(ctx, u)
			if err != nil {
				continue // skip failed fetches
			}
			results = append(results, content)
		}
		return results, nil
	default:
		return nil, fmt.Errorf("unsupported article type: %s", src.Type)
	}
}

func FetchArticleList(ctx context.Context, src config.ArticleSource) ([]string, error) {
	if src.Type != "url_list" {
		return nil, fmt.Errorf("FetchArticleList only supports url_list type")
	}

	f, err := os.Open(src.Path)
	if err != nil {
		return nil, fmt.Errorf("opening bookmark file: %w", err)
	}
	defer f.Close()

	var contents []string
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		url := strings.TrimSpace(scanner.Text())
		if url == "" || strings.HasPrefix(url, "#") {
			continue
		}
		content, err := fetchURL(ctx, url)
		if err != nil {
			return nil, fmt.Errorf("fetching %s: %w", url, err)
		}
		contents = append(contents, content)
	}
	return contents, scanner.Err()
}

func fetchURL(ctx context.Context, url string) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("User-Agent", "gocoder/1.0 (dataset collection)")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("HTTP %d for %s", resp.StatusCode, url)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	return string(body), nil
}
