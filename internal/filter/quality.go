package filter

import (
	"strings"

	"github.com/yavosh/gocoder/internal/extractor"
)

func HasMinLines(f extractor.ExtractedFunc, minLines int) bool {
	lines := strings.Count(f.Body, "\n") + 1
	return lines >= minLines
}

func FilterAll(funcs []extractor.ExtractedFunc, minLines int) []extractor.ExtractedFunc {
	var result []extractor.ExtractedFunc
	for _, f := range funcs {
		if IsGenerated(f) {
			continue
		}
		if !HasMinLines(f, minLines) {
			continue
		}
		result = append(result, f)
	}
	return Dedup(result)
}
