package formatter

import (
	"fmt"
	"math/rand"

	"github.com/yavosh/gocoder/internal/extractor"
)

const (
	FIMPrefix = "<|fim_prefix|>"
	FIMSuffix = "<|fim_suffix|>"
	FIMMiddle = "<|fim_middle|>"
)

func FormatFIM(f extractor.ExtractedFunc, seed int64) (TrainingExample, error) {
	body := f.Body
	if len(body) < 10 {
		return TrainingExample{}, fmt.Errorf("function too short for FIM")
	}

	rng := rand.New(rand.NewSource(seed))

	minPos := len(body) / 4
	maxPos := len(body) * 3 / 4
	if minPos >= maxPos {
		minPos = 1
		maxPos = len(body) - 1
	}
	splitPos := minPos + rng.Intn(maxPos-minPos)

	suffixStart := splitPos + (len(body)-splitPos)/2
	if suffixStart >= len(body) {
		suffixStart = splitPos
	}

	prefix := body[:splitPos]
	middle := body[splitPos:suffixStart]
	suffix := body[suffixStart:]

	text := fmt.Sprintf("%s%s%s%s%s%s", FIMPrefix, prefix, FIMSuffix, suffix, FIMMiddle, middle)

	return TrainingExample{
		Text:     text,
		Source:   f.FilePath,
		Category: "fim",
	}, nil
}
