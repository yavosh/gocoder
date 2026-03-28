package filter

import (
	"strings"

	"github.com/yavosh/gocoder/internal/extractor"
)

func IsGenerated(f extractor.ExtractedFunc) bool {
	s := f.Body
	if len(s) > 512 {
		s = s[:512]
	}
	return strings.Contains(s, "DO NOT EDIT") ||
		strings.Contains(s, "Code generated") ||
		strings.Contains(s, "go:generate")
}
