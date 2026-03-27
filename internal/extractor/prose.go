package extractor

import (
	"strings"

	"golang.org/x/net/html"
)

// ProseSection represents a section of an article with prose text and code blocks.
type ProseSection struct {
	Title      string
	Prose      string
	CodeBlocks []string
}

func ExtractProse(htmlContent string) ([]ProseSection, error) {
	doc, err := html.Parse(strings.NewReader(htmlContent))
	if err != nil {
		return nil, err
	}

	var sections []ProseSection
	var current ProseSection

	var walk func(*html.Node)
	walk = func(n *html.Node) {
		if n.Type == html.ElementNode {
			switch n.Data {
			case "h1", "h2", "h3":
				if current.Prose != "" || len(current.CodeBlocks) > 0 {
					sections = append(sections, current)
				}
				current = ProseSection{Title: textContent(n)}
			case "p":
				text := strings.TrimSpace(textContent(n))
				if text != "" {
					if current.Prose != "" {
						current.Prose += "\n"
					}
					current.Prose += text
				}
			case "pre":
				code := strings.TrimSpace(textContent(n))
				if code != "" {
					current.CodeBlocks = append(current.CodeBlocks, code)
				}
				return // don't recurse into code blocks
			}
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
	}
	walk(doc)

	if current.Prose != "" || len(current.CodeBlocks) > 0 {
		sections = append(sections, current)
	}

	return sections, nil
}

func textContent(n *html.Node) string {
	if n.Type == html.TextNode {
		return n.Data
	}
	var sb strings.Builder
	for c := n.FirstChild; c != nil; c = c.NextSibling {
		sb.WriteString(textContent(c))
	}
	return sb.String()
}
