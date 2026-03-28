package pipeline

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"

	"github.com/yavosh/gocoder/internal/extractor"
	"github.com/yavosh/gocoder/internal/filter"
	"github.com/yavosh/gocoder/internal/formatter"
)

type BuildOptions struct {
	MinLines  int
	FIMRatio  float64
	Seed      int64
	EvalSplit float64 // default 0.1
}

type BuildResult struct {
	TotalExamples int
	TrainExamples int
	EvalExamples  int
	FIMExamples   int
	InstrExamples int
}

func BuildFromDir(ctx context.Context, inputDir, outputDir string, opts BuildOptions) (*BuildResult, error) {
	if opts.EvalSplit == 0 {
		opts.EvalSplit = 0.1
	}

	// Find all Go files, skipping vendor/ directories
	var goFiles []string
	err := filepath.Walk(inputDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() && info.Name() == "vendor" {
			return filepath.SkipDir
		}
		if !info.IsDir() && filepath.Ext(path) == ".go" {
			goFiles = append(goFiles, path)
		}
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("walking input dir: %w", err)
	}

	// Extract functions from all files
	var allFuncs []extractor.ExtractedFunc
	for _, path := range goFiles {
		funcs, err := extractor.ExtractFunctions(path)
		if err != nil {
			continue // skip unparseable files
		}
		allFuncs = append(allFuncs, funcs...)
	}

	// Filter
	filtered := filter.FilterAll(allFuncs, opts.MinLines)

	// Format
	rng := rand.New(rand.NewSource(opts.Seed))
	var examples []formatter.TrainingExample

	for i, f := range filtered {
		if rng.Float64() < opts.FIMRatio {
			ex, err := formatter.FormatFIM(f, opts.Seed+int64(i))
			if err != nil {
				continue
			}
			examples = append(examples, ex)
		} else {
			examples = append(examples, formatter.FormatInstruction(f))
		}
	}

	if len(examples) == 0 {
		// Write empty files
		os.WriteFile(filepath.Join(outputDir, "train.jsonl"), nil, 0o644)
		os.WriteFile(filepath.Join(outputDir, "eval.jsonl"), nil, 0o644)
		return &BuildResult{}, nil
	}

	// Shuffle deterministically
	rng.Shuffle(len(examples), func(i, j int) {
		examples[i], examples[j] = examples[j], examples[i]
	})

	// Split
	evalCount := int(float64(len(examples)) * opts.EvalSplit)
	if evalCount < 1 {
		evalCount = 1
	}
	if evalCount >= len(examples) {
		evalCount = len(examples) - 1
	}
	evalExamples := examples[:evalCount]
	trainExamples := examples[evalCount:]

	// Write JSONL
	if err := writeJSONL(filepath.Join(outputDir, "train.jsonl"), trainExamples); err != nil {
		return nil, err
	}
	if err := writeJSONL(filepath.Join(outputDir, "eval.jsonl"), evalExamples); err != nil {
		return nil, err
	}

	fimCount := 0
	for _, e := range examples {
		if e.Text != "" {
			fimCount++
		}
	}

	return &BuildResult{
		TotalExamples: len(examples),
		TrainExamples: len(trainExamples),
		EvalExamples:  len(evalExamples),
		FIMExamples:   fimCount,
		InstrExamples: len(examples) - fimCount,
	}, nil
}

func writeJSONL(path string, examples []formatter.TrainingExample) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("creating %s: %w", path, err)
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	for _, ex := range examples {
		if err := enc.Encode(ex); err != nil {
			return fmt.Errorf("encoding example: %w", err)
		}
	}
	return nil
}
