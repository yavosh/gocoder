package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"github.com/yavosh/gocoder/internal/pipeline"
)

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	switch os.Args[1] {
	case "pipeline":
		runPipeline(os.Args[2:])
	case "eval":
		runEval(os.Args[2:])
	case "serve":
		runServe(os.Args[2:])
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n", os.Args[1])
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Fprintln(os.Stderr, "Usage: gocoder <command> [args]")
	fmt.Fprintln(os.Stderr, "")
	fmt.Fprintln(os.Stderr, "Commands:")
	fmt.Fprintln(os.Stderr, "  pipeline build    Build training dataset from sources")
	fmt.Fprintln(os.Stderr, "  eval run          Run evaluation prompts against a model")
	fmt.Fprintln(os.Stderr, "  eval compare      Compare two evaluation runs")
	fmt.Fprintln(os.Stderr, "  eval judge        Run LLM-as-judge on results")
	fmt.Fprintln(os.Stderr, "  serve             Start model routing proxy")
}

func runPipeline(args []string) {
	if len(args) < 1 || args[0] != "build" {
		fmt.Fprintln(os.Stderr, "Usage: gocoder pipeline build [flags]")
		os.Exit(1)
	}

	fs := flag.NewFlagSet("pipeline", flag.ExitOnError)
	dir := fs.String("dir", ".", "Input directory containing Go source files")
	output := fs.String("output", "data/output", "Output directory for JSONL files")
	minLines := fs.Int("min-lines", 3, "Minimum function body lines")
	fimRatio := fs.Float64("fim-ratio", 0.4, "Fraction of examples formatted as FIM")
	seed := fs.Int64("seed", 42, "Random seed for reproducibility")
	evalSplit := fs.Float64("eval-split", 0.1, "Fraction of examples for eval set")
	fs.Parse(args[1:])

	if err := os.MkdirAll(*output, 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "creating output dir: %v\n", err)
		os.Exit(1)
	}

	result, err := pipeline.BuildFromDir(context.Background(), *dir, *output, pipeline.BuildOptions{
		MinLines:  *minLines,
		FIMRatio:  *fimRatio,
		Seed:      *seed,
		EvalSplit: *evalSplit,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "pipeline build: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Pipeline build complete:\n")
	fmt.Printf("  Total examples: %d\n", result.TotalExamples)
	fmt.Printf("  Train:          %d\n", result.TrainExamples)
	fmt.Printf("  Eval:           %d\n", result.EvalExamples)
	fmt.Printf("  FIM:            %d\n", result.FIMExamples)
	fmt.Printf("  Instruction:    %d\n", result.InstrExamples)
}

func runEval(_ []string) {
	fmt.Fprintln(os.Stderr, "eval: not implemented")
	os.Exit(1)
}

func runServe(_ []string) {
	fmt.Fprintln(os.Stderr, "serve: not implemented")
	os.Exit(1)
}
