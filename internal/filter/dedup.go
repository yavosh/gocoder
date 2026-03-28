package filter

import (
	"crypto/sha256"
	"encoding/hex"

	"github.com/yavosh/gocoder/internal/extractor"
)

func Dedup(funcs []extractor.ExtractedFunc) []extractor.ExtractedFunc {
	seen := make(map[string]bool)
	var result []extractor.ExtractedFunc

	for _, f := range funcs {
		h := hash(f.Body)
		if seen[h] {
			continue
		}
		seen[h] = true
		result = append(result, f)
	}
	return result
}

func hash(s string) string {
	h := sha256.Sum256([]byte(s))
	return hex.EncodeToString(h[:])
}
