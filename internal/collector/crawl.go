package collector

import (
	"context"
	"encoding/xml"
	"fmt"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/yavosh/gocoder/internal/config"
	"golang.org/x/net/html"
)

// DiscoverBlogURLs fetches a blog index page and extracts article links
// that share the same host as the index URL.
func DiscoverBlogURLs(ctx context.Context, indexURL string) ([]string, error) {
	body, err := fetchURL(ctx, indexURL)
	if err != nil {
		return nil, err
	}

	base, err := url.Parse(indexURL)
	if err != nil {
		return nil, err
	}

	doc, err := html.Parse(strings.NewReader(body))
	if err != nil {
		return nil, err
	}

	var urls []string
	seen := make(map[string]bool)

	var walk func(*html.Node)
	walk = func(n *html.Node) {
		if n.Type == html.ElementNode && n.Data == "a" {
			for _, attr := range n.Attr {
				if attr.Key == "href" {
					resolved := resolveURL(base, attr.Val)
					if resolved != "" && !seen[resolved] && strings.HasPrefix(resolved, base.Scheme+"://"+base.Host) {
						seen[resolved] = true
						urls = append(urls, resolved)
					}
				}
			}
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
	}
	walk(doc)

	return urls, nil
}

func resolveURL(base *url.URL, href string) string {
	ref, err := url.Parse(href)
	if err != nil {
		return ""
	}
	return base.ResolveReference(ref).String()
}

// FetchMarkdownRepo clones a git repo and returns the content of all .md files.
func FetchMarkdownRepo(ctx context.Context, src config.ArticleSource) ([]string, error) {
	dir, err := os.MkdirTemp("", "gocoder-md-*")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(dir)

	parts := strings.Split(strings.TrimSuffix(src.URL, "/"), "/")
	name := parts[len(parts)-1]
	dest := filepath.Join(dir, name)

	cmd := exec.CommandContext(ctx, "git", "clone", "--depth", "1", src.URL, dest)
	if out, err := cmd.CombinedOutput(); err != nil {
		return nil, fmt.Errorf("git clone %s: %w\n%s", src.URL, err, out)
	}

	var contents []string
	filepath.Walk(dest, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && strings.HasSuffix(path, ".md") {
			data, err := os.ReadFile(path)
			if err == nil && len(data) > 0 {
				contents = append(contents, string(data))
			}
		}
		return nil
	})

	return contents, nil
}

type sitemapURLSet struct {
	URLs []sitemapURL `xml:"url"`
}

type sitemapURL struct {
	Loc string `xml:"loc"`
}

// FetchSitemapURLs parses a sitemap.xml and returns all URLs.
func FetchSitemapURLs(ctx context.Context, sitemapURL string) ([]string, error) {
	body, err := fetchURL(ctx, sitemapURL)
	if err != nil {
		return nil, err
	}

	var urlset sitemapURLSet
	if err := xml.Unmarshal([]byte(body), &urlset); err != nil {
		return nil, fmt.Errorf("parsing sitemap: %w", err)
	}

	var urls []string
	for _, u := range urlset.URLs {
		if u.Loc != "" {
			urls = append(urls, u.Loc)
		}
	}
	return urls, nil
}
