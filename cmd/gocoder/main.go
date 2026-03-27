package main

import (
	"fmt"
	"os"
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

func runPipeline(_ []string) {
	fmt.Fprintln(os.Stderr, "pipeline: not implemented")
	os.Exit(1)
}

func runEval(_ []string) {
	fmt.Fprintln(os.Stderr, "eval: not implemented")
	os.Exit(1)
}

func runServe(_ []string) {
	fmt.Fprintln(os.Stderr, "serve: not implemented")
	os.Exit(1)
}
