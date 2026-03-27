package extractor_test

import (
	"testing"

	"github.com/yavosh/gocoder/internal/extractor"
)

func TestExtractProse(t *testing.T) {
	html := `<h2>Error Handling</h2>
<p>In Go, errors are values. The idiomatic way to handle them is:</p>
<pre><code>if err != nil {
    return fmt.Errorf("operation failed: %w", err)
}</code></pre>
<p>Always wrap errors with context using fmt.Errorf.</p>`

	sections, err := extractor.ExtractProse(html)
	if err != nil {
		t.Fatalf("ExtractProse: %v", err)
	}

	if len(sections) == 0 {
		t.Fatal("expected at least one section")
	}

	s := sections[0]
	if s.Prose == "" {
		t.Error("expected prose text")
	}
	if len(s.CodeBlocks) == 0 {
		t.Error("expected at least one code block")
	}
	if s.CodeBlocks[0] == "" {
		t.Error("expected non-empty code block")
	}
}
