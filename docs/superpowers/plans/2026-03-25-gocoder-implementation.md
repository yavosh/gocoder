# GoCoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Go-specialized LLM fine-tuning pipeline, evaluation harness, and local serving stack around Nemotron-Cascade-2.

**Architecture:** Unified `gocoder` Go CLI with subcommands (`pipeline build`, `eval run`, `serve`). Python scripts for ML training only. All Go components share a single go.mod at the repo root. The router is a thin OpenAI-compatible proxy.

**Tech Stack:** Go 1.22+, Python 3.11+ (unsloth, trl, peft, transformers), ollama, YAML config, JSONL data format.

**Spec:** `docs/superpowers/specs/2026-03-25-gocoder-design.md`

---

## File Structure

```
gocoder/
├── go.mod
├── go.sum
├── Makefile
├── cmd/
│   └── gocoder/
│       └── main.go                     # CLI entrypoint with subcommands
├── internal/
│   ├── config/
│   │   └── sources.go                  # Parse sources.yaml
│   ├── collector/
│   │   ├── repo.go                     # Clone git repos
│   │   └── article.go                  # Fetch articles/blogs as markdown
│   ├── extractor/
│   │   ├── gocode.go                   # AST extraction via go/parser
│   │   └── prose.go                    # Article HTML -> markdown, extract code blocks
│   ├── filter/
│   │   ├── generated.go                # Remove generated code
│   │   ├── dedup.go                    # MinHash near-duplicate removal
│   │   └── quality.go                  # go vet gate, min lines
│   ├── formatter/
│   │   ├── fim.go                      # FIM format conversion
│   │   └── instruction.go             # Instruction format conversion
│   ├── eval/
│   │   ├── runner.go                   # Run prompts against model API
│   │   ├── scorer.go                   # Automated scoring (compile, vet, staticcheck)
│   │   ├── judge.go                    # LLM-as-judge scoring
│   │   └── compare.go                 # Compare two result sets
│   └── router/
│       ├── proxy.go                    # HTTP reverse proxy with model routing
│       ├── config.go                   # Parse serving config.yaml
│       └── metrics.go                  # Prometheus metrics
├── data/
│   ├── sources.yaml
│   └── .gitignore
├── eval/
│   └── prompts/                        # YAML prompt files by category
│       ├── error_handling.yaml
│       ├── concurrency.yaml
│       ├── http_middleware.yaml
│       ├── interfaces.yaml
│       ├── testing.yaml
│       ├── context_propagation.yaml
│       ├── package_design.yaml
│       ├── struct_methods.yaml
│       └── idiom.yaml
├── serving/
│   ├── config.yaml
│   └── models/
│       └── go-nemotron.Modelfile
├── training/
│   ├── train.py
│   ├── merge_lora.py
│   ├── convert_gguf.py
│   ├── config/
│   │   └── nemotron-cascade-2.yaml
│   └── requirements.txt
└── results/                            # Gitignored eval results
    └── .gitignore
```

---

## Task 1: Project Scaffold & CLI Skeleton

**Files:**
- Create: `go.mod`
- Create: `cmd/gocoder/main.go`
- Create: `Makefile`
- Create: `data/.gitignore`
- Create: `results/.gitignore`

- [ ] **Step 1: Initialize Go module**

```bash
cd /Users/yavosh/Projects/yavosh/gocoder
go mod init github.com/yavosh/gocoder
```

- [ ] **Step 2: Write CLI entrypoint with subcommand stubs**

Create `cmd/gocoder/main.go`:

```go
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

func runPipeline(args []string) {
	fmt.Fprintln(os.Stderr, "pipeline: not implemented")
	os.Exit(1)
}

func runEval(args []string) {
	fmt.Fprintln(os.Stderr, "eval: not implemented")
	os.Exit(1)
}

func runServe(args []string) {
	fmt.Fprintln(os.Stderr, "serve: not implemented")
	os.Exit(1)
}
```

- [ ] **Step 3: Write Makefile**

Create `Makefile`:

```makefile
.PHONY: build test lint clean

build:
	go build -o bin/gocoder ./cmd/gocoder

test:
	go test ./...

lint:
	go vet ./...
	staticcheck ./...

clean:
	rm -rf bin/
```

- [ ] **Step 4: Create gitignore files**

Create `data/.gitignore`:

```
# Keep sources.yaml, ignore everything else
*
!.gitignore
!sources.yaml
!bookmarks.txt
```

Create `results/.gitignore`:

```
*
!.gitignore
```

- [ ] **Step 5: Create root .gitignore**

Create `.gitignore`:

```
bin/
```

- [ ] **Step 6: Write test for CLI dispatch**

Create `cmd/gocoder/main_test.go`:

```go
package main

import (
	"os"
	"os/exec"
	"testing"
)

func TestCLINoArgs(t *testing.T) {
	cmd := exec.Command("go", "run", ".", )
	cmd.Dir = "."
	err := cmd.Run()
	if err == nil {
		t.Error("expected non-zero exit with no args")
	}
}

func TestCLIUnknownCommand(t *testing.T) {
	cmd := exec.Command("go", "run", ".", "bogus")
	cmd.Dir = "."
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Error("expected non-zero exit for unknown command")
	}
	if !contains(string(out), "unknown command") {
		t.Errorf("expected 'unknown command' in output, got: %s", out)
	}
}

func TestCLIPipelineStub(t *testing.T) {
	cmd := exec.Command("go", "run", ".", "pipeline", "build")
	cmd.Dir = "."
	out, _ := cmd.CombinedOutput()
	if !contains(string(out), "not implemented") {
		t.Errorf("expected 'not implemented', got: %s", out)
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && containsHelper(s, substr))
}

func containsHelper(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

func TestMain(m *testing.M) {
	os.Exit(m.Run())
}
```

- [ ] **Step 7: Run tests**

```bash
go test ./cmd/gocoder/...
```

Expected: PASS.

- [ ] **Step 8: Verify build**

```bash
make build
bin/gocoder
```

Expected: prints usage and exits with code 1.

```bash
bin/gocoder pipeline build
```

Expected: prints "pipeline: not implemented" and exits with code 1.

- [ ] **Step 9: Commit**

```bash
git add go.mod cmd/ Makefile data/.gitignore results/.gitignore .gitignore
git commit -m "feat: scaffold gocoder CLI with subcommand stubs"
```

---

## Task 2: Sources Config Parser

**Files:**
- Create: `internal/config/sources.go`
- Create: `internal/config/sources_test.go`
- Create: `data/sources.yaml`

- [ ] **Step 1: Write test for sources config parsing**

Create `internal/config/sources_test.go`:

```go
package config_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/yavosh/gocoder/internal/config"
)

func TestParseSourcesYAML(t *testing.T) {
	dir := t.TempDir()
	yamlContent := `repos:
  - url: https://github.com/golang/go
    path: src/
    weight: 1.5
  - url: https://github.com/cockroachdb/cockroach
articles:
  - url: https://go.dev/blog/
    type: blog_index
  - path: data/bookmarks.txt
    type: url_list
`
	path := filepath.Join(dir, "sources.yaml")
	if err := os.WriteFile(path, []byte(yamlContent), 0o644); err != nil {
		t.Fatal(err)
	}

	cfg, err := config.ParseSources(path)
	if err != nil {
		t.Fatalf("ParseSources: %v", err)
	}

	if len(cfg.Repos) != 2 {
		t.Fatalf("expected 2 repos, got %d", len(cfg.Repos))
	}
	if cfg.Repos[0].Path != "src/" {
		t.Errorf("expected path 'src/', got %q", cfg.Repos[0].Path)
	}
	if cfg.Repos[0].Weight != 1.5 {
		t.Errorf("expected weight 1.5, got %f", cfg.Repos[0].Weight)
	}
	if cfg.Repos[1].Weight != 0 {
		t.Errorf("expected default weight 0, got %f", cfg.Repos[1].Weight)
	}
	if len(cfg.Articles) != 2 {
		t.Fatalf("expected 2 articles, got %d", len(cfg.Articles))
	}
	if cfg.Articles[0].Type != "blog_index" {
		t.Errorf("expected type 'blog_index', got %q", cfg.Articles[0].Type)
	}
	if cfg.Articles[1].Path != "data/bookmarks.txt" {
		t.Errorf("expected path 'data/bookmarks.txt', got %q", cfg.Articles[1].Path)
	}
}

func TestParseSourcesDefaultWeight(t *testing.T) {
	dir := t.TempDir()
	yamlContent := `repos:
  - url: https://github.com/golang/go
`
	path := filepath.Join(dir, "sources.yaml")
	if err := os.WriteFile(path, []byte(yamlContent), 0o644); err != nil {
		t.Fatal(err)
	}

	cfg, err := config.ParseSources(path)
	if err != nil {
		t.Fatalf("ParseSources: %v", err)
	}

	if cfg.Repos[0].EffectiveWeight() != 1.0 {
		t.Errorf("expected effective weight 1.0, got %f", cfg.Repos[0].EffectiveWeight())
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
go test ./internal/config/...
```

Expected: FAIL — package doesn't exist yet.

- [ ] **Step 3: Implement sources config parser**

Create `internal/config/sources.go`:

```go
package config

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

type Sources struct {
	Repos    []RepoSource    `yaml:"repos"`
	Articles []ArticleSource `yaml:"articles"`
}

type RepoSource struct {
	URL    string  `yaml:"url"`
	Path   string  `yaml:"path"`
	Weight float64 `yaml:"weight"`
}

func (r RepoSource) EffectiveWeight() float64 {
	if r.Weight == 0 {
		return 1.0
	}
	return r.Weight
}

type ArticleSource struct {
	URL  string `yaml:"url"`
	Path string `yaml:"path"`
	Type string `yaml:"type"`
}

func ParseSources(path string) (*Sources, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading sources: %w", err)
	}

	var s Sources
	if err := yaml.Unmarshal(data, &s); err != nil {
		return nil, fmt.Errorf("parsing sources: %w", err)
	}

	return &s, nil
}
```

- [ ] **Step 4: Add yaml dependency and run tests**

```bash
go get gopkg.in/yaml.v3 && go test ./internal/config/...
```

Expected: PASS.

- [ ] **Step 5: Create the actual sources.yaml**

Create `data/sources.yaml`:

```yaml
repos:
  - url: https://github.com/golang/go
    path: src/
    weight: 1.5
  - url: https://github.com/cockroachdb/cockroach
  - url: https://github.com/kubernetes/kubernetes
  - url: https://github.com/prometheus/prometheus
  - url: https://github.com/hashicorp/consul
  - url: https://github.com/hashicorp/vault
  - url: https://github.com/hashicorp/terraform
  - url: https://github.com/etcd-io/etcd
  - url: https://github.com/charmbracelet/bubbletea
  - url: https://github.com/sqlc-dev/sqlc
  - url: https://github.com/coredns/coredns

  # Personal repos (update with actual URLs)
  # - url: https://github.com/yavosh/repo1
  #   weight: 2.0

articles:
  - url: https://go.dev/blog/
    type: blog_index
  - url: https://go.dev/doc/effective_go
    type: single_page
  - url: https://github.com/uber-go/guide
    type: markdown_repo
  - url: https://google.github.io/styleguide/go/
    type: sitemap
  - url: https://dave.cheney.net/
    type: blog_index
  - url: https://eli.thegreenplace.net/tag/go
    type: blog_index
  - path: data/bookmarks.txt
    type: url_list
```

- [ ] **Step 6: Commit**

```bash
git add internal/config/ data/sources.yaml go.mod go.sum
git commit -m "feat: add sources.yaml config parser"
```

---

## Task 3: Repo Collector

**Files:**
- Create: `internal/collector/repo.go`
- Create: `internal/collector/repo_test.go`

- [ ] **Step 1: Write test for repo cloning**

Create `internal/collector/repo_test.go`:

```go
package collector_test

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/yavosh/gocoder/internal/collector"
	"github.com/yavosh/gocoder/internal/config"
)

func TestCloneRepo(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test")
	}

	dir := t.TempDir()
	src := config.RepoSource{
		URL: "https://github.com/charmbracelet/bubbletea",
	}

	repoDir, err := collector.CloneRepo(context.Background(), src, dir)
	if err != nil {
		t.Fatalf("CloneRepo: %v", err)
	}

	// Verify Go files exist
	goFiles, err := filepath.Glob(filepath.Join(repoDir, "*.go"))
	if err != nil {
		t.Fatal(err)
	}
	if len(goFiles) == 0 {
		t.Error("expected Go files in cloned repo")
	}
}

func TestCloneRepoWithPath(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test")
	}

	dir := t.TempDir()
	src := config.RepoSource{
		URL:  "https://github.com/charmbracelet/bubbletea",
		Path: "examples/",
	}

	repoDir, err := collector.CloneRepo(context.Background(), src, dir)
	if err != nil {
		t.Fatalf("CloneRepo: %v", err)
	}

	// When Path is set, repoDir should point to the subpath
	if info, err := os.Stat(repoDir); err != nil || !info.IsDir() {
		t.Errorf("expected repoDir to be a directory at examples/, got error: %v", err)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
go test ./internal/collector/... -short
```

Expected: FAIL — package doesn't exist.

- [ ] **Step 3: Implement repo collector**

Create `internal/collector/repo.go`:

```go
package collector

import (
	"context"
	"fmt"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/yavosh/gocoder/internal/config"
)

func CloneRepo(ctx context.Context, src config.RepoSource, outputDir string) (string, error) {
	// Derive directory name from URL
	parts := strings.Split(strings.TrimSuffix(src.URL, "/"), "/")
	name := parts[len(parts)-1]

	dest := filepath.Join(outputDir, name)

	cmd := exec.CommandContext(ctx, "git", "clone", "--depth", "1", src.URL, dest)
	if out, err := cmd.CombinedOutput(); err != nil {
		return "", fmt.Errorf("git clone %s: %w\n%s", src.URL, err, out)
	}

	if src.Path != "" {
		return filepath.Join(dest, src.Path), nil
	}
	return dest, nil
}

func CloneAll(ctx context.Context, repos []config.RepoSource, outputDir string) ([]string, error) {
	var dirs []string
	for _, r := range repos {
		dir, err := CloneRepo(ctx, r, outputDir)
		if err != nil {
			return nil, err
		}
		dirs = append(dirs, dir)
	}
	return dirs, nil
}
```

- [ ] **Step 4: Run tests**

```bash
go test ./internal/collector/... -run TestCloneRepo -count=1
```

Expected: PASS (requires network, ~10-30 seconds).

- [ ] **Step 5: Commit**

```bash
git add internal/collector/
git commit -m "feat: add git repo collector"
```

---

## Task 4: Go Code Extractor (AST-based)

**Files:**
- Create: `internal/extractor/gocode.go`
- Create: `internal/extractor/gocode_test.go`
- Create: `internal/extractor/types.go`

- [ ] **Step 1: Define extracted code types**

Create `internal/extractor/types.go`:

```go
package extractor

type ExtractedFunc struct {
	Package    string
	Name       string
	Receiver   string   // empty if not a method
	Doc        string   // godoc comment
	Signature  string   // full signature line
	Body       string   // full function source
	Imports    []string // imports used by this function
	FilePath   string
	LineNumber int
}
```

- [ ] **Step 2: Write test for Go code extraction**

Create `internal/extractor/gocode_test.go`:

```go
package extractor_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/yavosh/gocoder/internal/extractor"
)

const testGoFile = `package server

import (
	"context"
	"fmt"
)

// HandleRequest processes an incoming request and returns an error
// if validation fails.
func (s *Server) HandleRequest(ctx context.Context, req *Request) error {
	if err := s.validate(req); err != nil {
		return fmt.Errorf("validation: %w", err)
	}
	return nil
}

// NewServer creates a new Server with the given config.
func NewServer(cfg Config) *Server {
	return &Server{cfg: cfg}
}
`

func TestExtractFunctions(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "server.go")
	if err := os.WriteFile(path, []byte(testGoFile), 0o644); err != nil {
		t.Fatal(err)
	}

	funcs, err := extractor.ExtractFunctions(path)
	if err != nil {
		t.Fatalf("ExtractFunctions: %v", err)
	}

	if len(funcs) != 2 {
		t.Fatalf("expected 2 functions, got %d", len(funcs))
	}

	// Check HandleRequest
	hr := funcs[0]
	if hr.Name != "HandleRequest" {
		t.Errorf("expected name HandleRequest, got %q", hr.Name)
	}
	if hr.Receiver != "*Server" {
		t.Errorf("expected receiver *Server, got %q", hr.Receiver)
	}
	if hr.Doc == "" {
		t.Error("expected doc comment, got empty")
	}
	if hr.Package != "server" {
		t.Errorf("expected package server, got %q", hr.Package)
	}

	// Check NewServer
	ns := funcs[1]
	if ns.Name != "NewServer" {
		t.Errorf("expected name NewServer, got %q", ns.Name)
	}
	if ns.Receiver != "" {
		t.Errorf("expected no receiver, got %q", ns.Receiver)
	}
}

func TestExtractFunctionsSkipsGenerated(t *testing.T) {
	dir := t.TempDir()
	content := "// Code generated by protoc-gen-go. DO NOT EDIT.\npackage pb\n\nfunc Foo() {}\n"
	path := filepath.Join(dir, "gen.pb.go")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	funcs, err := extractor.ExtractFunctions(path)
	if err != nil {
		t.Fatalf("ExtractFunctions: %v", err)
	}
	if len(funcs) != 0 {
		t.Errorf("expected 0 functions from generated file, got %d", len(funcs))
	}
}
```

- [ ] **Step 3: Run test to verify it fails**

```bash
go test ./internal/extractor/...
```

Expected: FAIL — ExtractFunctions not defined.

- [ ] **Step 4: Implement Go code extractor**

Create `internal/extractor/gocode.go`:

```go
package extractor

import (
	"bytes"
	"go/ast"
	"go/parser"
	"go/printer"
	"go/token"
	"os"
	"strings"
)

// IsGeneratedFile checks if file-level header indicates generated code.
// The filter package handles per-function checks — this is file-level only.
func IsGeneratedFile(src []byte) bool {
	first512 := src
	if len(first512) > 512 {
		first512 = first512[:512]
	}
	s := string(first512)
	return strings.Contains(s, "DO NOT EDIT") ||
		strings.Contains(s, "Code generated")
}

func ExtractFunctions(path string) ([]ExtractedFunc, error) {
	src, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	// Skip generated files at file level
	if IsGeneratedFile(src) {
		return nil, nil
	}

	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, path, src, parser.ParseComments)
	if err != nil {
		return nil, err
	}

	var funcs []ExtractedFunc
	for _, decl := range f.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok {
			continue
		}

		ef := ExtractedFunc{
			Package:    f.Name.Name,
			Name:       fn.Name.Name,
			FilePath:   path,
			LineNumber: fset.Position(fn.Pos()).Line,
		}

		// Receiver
		if fn.Recv != nil && len(fn.Recv.List) > 0 {
			var buf bytes.Buffer
			printer.Fprint(&buf, fset, fn.Recv.List[0].Type)
			ef.Receiver = buf.String()
		}

		// Doc comment
		if fn.Doc != nil {
			ef.Doc = strings.TrimSpace(fn.Doc.Text())
		}

		// Full function body
		var bodyBuf bytes.Buffer
		printer.Fprint(&bodyBuf, fset, fn)
		ef.Body = bodyBuf.String()

		// Signature (first line of body)
		lines := strings.SplitN(ef.Body, "\n", 2)
		ef.Signature = strings.TrimSuffix(lines[0], " {")

		// Imports
		for _, imp := range f.Imports {
			ef.Imports = append(ef.Imports, strings.Trim(imp.Path.Value, `"`))
		}

		funcs = append(funcs, ef)
	}

	return funcs, nil
}

// Note: isGenerated removed — use IsGeneratedFile for file-level check,
// filter.IsGenerated for function-level check. Single source of truth.
```

- [ ] **Step 5: Run tests**

```bash
go test ./internal/extractor/...
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add internal/extractor/
git commit -m "feat: add Go AST-based code extractor"
```

---

## Task 5: Article Collector & Prose Extractor

**Files:**
- Create: `internal/collector/article.go`
- Create: `internal/collector/article_test.go`
- Create: `internal/extractor/prose.go`
- Create: `internal/extractor/prose_test.go`

- [ ] **Step 1: Write test for article fetching**

Create `internal/collector/article_test.go` — test that fetching a single page returns HTML content. Use a short test that hits a known stable URL.

```go
package collector_test

import (
	"context"
	"testing"

	"github.com/yavosh/gocoder/internal/collector"
	"github.com/yavosh/gocoder/internal/config"
)

func TestFetchArticle(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test")
	}

	src := config.ArticleSource{
		URL:  "https://go.dev/doc/effective_go",
		Type: "single_page",
	}

	content, err := collector.FetchArticle(context.Background(), src)
	if err != nil {
		t.Fatalf("FetchArticle: %v", err)
	}
	if len(content) == 0 {
		t.Error("expected non-empty content")
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
go test ./internal/collector/... -short
```

Expected: FAIL — FetchArticle not defined.

- [ ] **Step 3: Implement article collector**

Create `internal/collector/article.go` — HTTP fetch for single pages. Blog index and sitemap crawling can be added incrementally (start with `single_page` and `url_list` types).

```go
package collector

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/yavosh/gocoder/internal/config"
)

func FetchArticle(ctx context.Context, src config.ArticleSource) (string, error) {
	switch src.Type {
	case "single_page":
		return fetchURL(ctx, src.URL)
	case "url_list":
		return "", fmt.Errorf("url_list type requires FetchArticles, not FetchArticle")
	default:
		return "", fmt.Errorf("unsupported article type: %s", src.Type)
	}
}

func FetchArticleList(ctx context.Context, src config.ArticleSource) ([]string, error) {
	if src.Type != "url_list" {
		return nil, fmt.Errorf("FetchArticleList only supports url_list type")
	}

	f, err := os.Open(src.Path)
	if err != nil {
		return nil, fmt.Errorf("opening bookmark file: %w", err)
	}
	defer f.Close()

	var contents []string
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		url := strings.TrimSpace(scanner.Text())
		if url == "" || strings.HasPrefix(url, "#") {
			continue
		}
		content, err := fetchURL(ctx, url)
		if err != nil {
			return nil, fmt.Errorf("fetching %s: %w", url, err)
		}
		contents = append(contents, content)
	}
	return contents, scanner.Err()
}

func fetchURL(ctx context.Context, url string) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("User-Agent", "gocoder/1.0 (dataset collection)")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("HTTP %d for %s", resp.StatusCode, url)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	return string(body), nil
}
```

- [ ] **Step 4: Write test for prose extraction (HTML to markdown with code blocks)**

Create `internal/extractor/prose_test.go`:

```go
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
```

- [ ] **Step 5: Implement prose extractor**

Create `internal/extractor/prose.go` — parses HTML, extracts text paragraphs and code blocks, groups them into sections.

```go
package extractor

import (
	"strings"

	"golang.org/x/net/html"
)

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
			case "pre", "code":
				if n.Parent != nil && n.Parent.Data == "pre" || n.Data == "pre" {
					code := strings.TrimSpace(textContent(n))
					if code != "" {
						current.CodeBlocks = append(current.CodeBlocks, code)
					}
					return // don't recurse into code blocks
				}
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
```

- [ ] **Step 6: Add dependency and run all tests**

```bash
go get golang.org/x/net && go test ./internal/collector/... ./internal/extractor/... -short
```

Expected: PASS (short mode skips network tests).

- [ ] **Step 7: Commit**

```bash
git add internal/collector/article.go internal/collector/article_test.go internal/extractor/prose.go internal/extractor/prose_test.go go.mod go.sum
git commit -m "feat: add article collector and prose extractor"
```

---

## Task 6: Filters (Generated Code, Dedup, Quality)

**Files:**
- Create: `internal/filter/generated.go`
- Create: `internal/filter/dedup.go`
- Create: `internal/filter/quality.go`
- Create: `internal/filter/filter_test.go`

- [ ] **Step 1: Write tests for all three filters**

Create `internal/filter/filter_test.go`:

```go
package filter_test

import (
	"testing"

	"github.com/yavosh/gocoder/internal/extractor"
	"github.com/yavosh/gocoder/internal/filter"
)

func TestIsGenerated(t *testing.T) {
	tests := []struct {
		name     string
		body     string
		expected bool
	}{
		{"protobuf", "// Code generated by protoc-gen-go. DO NOT EDIT.\nfunc Foo() {}", true},
		{"mockgen", "// Code generated by MockGen. DO NOT EDIT.\nfunc Foo() {}", true},
		{"normal", "// HandleRequest processes a request.\nfunc HandleRequest() {}", false},
		{"wire", "//go:generate wire\nfunc Foo() {}", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := extractor.ExtractedFunc{Body: tt.body}
			if got := filter.IsGenerated(f); got != tt.expected {
				t.Errorf("IsGenerated(%q) = %v, want %v", tt.name, got, tt.expected)
			}
		})
	}
}

func TestDedup(t *testing.T) {
	funcs := []extractor.ExtractedFunc{
		{Name: "Foo", Body: "func Foo() { return 1 }"},
		{Name: "Foo", Body: "func Foo() { return 1 }"},   // exact dup
		{Name: "Bar", Body: "func Bar() { return 2 }"},
	}

	deduped := filter.Dedup(funcs)
	if len(deduped) != 2 {
		t.Errorf("expected 2 after dedup, got %d", len(deduped))
	}
}

func TestMinLines(t *testing.T) {
	short := extractor.ExtractedFunc{Body: "func F() {}"}
	long := extractor.ExtractedFunc{Body: "func F() {\n\ta := 1\n\tb := 2\n\tc := 3\n\treturn a + b + c\n}"}

	if filter.HasMinLines(short, 3) {
		t.Error("expected short function to fail min lines check")
	}
	if !filter.HasMinLines(long, 3) {
		t.Error("expected long function to pass min lines check")
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
go test ./internal/filter/...
```

Expected: FAIL — package doesn't exist.

- [ ] **Step 3: Implement filters**

Create `internal/filter/generated.go`:

```go
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
```

Create `internal/filter/dedup.go`:

```go
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
```

Create `internal/filter/quality.go`:

```go
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
```

- [ ] **Step 4: Run tests**

```bash
go test ./internal/filter/...
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add internal/filter/
git commit -m "feat: add generated code, dedup, and quality filters"
```

---

## Task 7: FIM & Instruction Formatters

**Files:**
- Create: `internal/formatter/fim.go`
- Create: `internal/formatter/instruction.go`
- Create: `internal/formatter/types.go`
- Create: `internal/formatter/formatter_test.go`

- [ ] **Step 1: Define training example types**

Create `internal/formatter/types.go`:

```go
package formatter

type TrainingExample struct {
	// For FIM format
	Text string `json:"text,omitempty"`

	// For instruction format
	Instruction string `json:"instruction,omitempty"`
	Output      string `json:"output,omitempty"`

	// Metadata
	Source   string `json:"source,omitempty"`
	Category string `json:"category,omitempty"`
}
```

- [ ] **Step 2: Write tests for FIM and instruction formatting**

Create `internal/formatter/formatter_test.go`:

```go
package formatter_test

import (
	"strings"
	"testing"

	"github.com/yavosh/gocoder/internal/extractor"
	"github.com/yavosh/gocoder/internal/formatter"
)

func TestFormatFIM(t *testing.T) {
	f := extractor.ExtractedFunc{
		Body: "func Add(a, b int) int {\n\treturn a + b\n}",
	}

	example, err := formatter.FormatFIM(f, 42) // deterministic seed
	if err != nil {
		t.Fatalf("FormatFIM: %v", err)
	}

	if example.Text == "" {
		t.Fatal("expected non-empty FIM text")
	}

	// Must contain FIM tokens
	for _, token := range []string{"<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"} {
		if !strings.Contains(example.Text, token) {
			t.Errorf("expected FIM text to contain %s", token)
		}
	}
}

func TestFormatInstruction(t *testing.T) {
	f := extractor.ExtractedFunc{
		Doc:       "Add returns the sum of a and b.",
		Signature: "func Add(a, b int) int",
		Body:      "func Add(a, b int) int {\n\treturn a + b\n}",
	}

	example := formatter.FormatInstruction(f)

	if example.Instruction == "" {
		t.Error("expected non-empty instruction")
	}
	if example.Output == "" {
		t.Error("expected non-empty output")
	}
	if !strings.Contains(example.Instruction, "Add returns") {
		t.Error("instruction should contain doc comment")
	}
}

func TestFormatProseInstruction(t *testing.T) {
	section := extractor.ProseSection{
		Prose:      "In Go, errors are values. Always wrap with context.",
		CodeBlocks: []string{"if err != nil {\n\treturn fmt.Errorf(\"op: %w\", err)\n}"},
	}

	examples := formatter.FormatProseInstruction(section)
	if len(examples) == 0 {
		t.Fatal("expected at least one example from prose section")
	}
}
```

- [ ] **Step 3: Run test to verify it fails**

```bash
go test ./internal/formatter/...
```

Expected: FAIL.

- [ ] **Step 4: Implement formatters**

Create `internal/formatter/fim.go`:

```go
package formatter

import (
	"fmt"
	"math/rand"

	"github.com/yavosh/gocoder/internal/extractor"
)

// FIM token constants — these are placeholders.
// Must be verified against the actual model's special tokens.
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

	// Pick a random split point (not at the very start or end)
	minPos := len(body) / 4
	maxPos := len(body) * 3 / 4
	if minPos >= maxPos {
		minPos = 1
		maxPos = len(body) - 1
	}
	splitPos := minPos + rng.Intn(maxPos-minPos)

	prefix := body[:splitPos]
	middle := body[splitPos:]

	// Find a second split for suffix (take some from the end)
	suffixStart := splitPos + len(middle)/2
	if suffixStart >= len(body) {
		suffixStart = splitPos
	}

	actualMiddle := body[splitPos:suffixStart]
	suffix := body[suffixStart:]

	text := fmt.Sprintf("%s%s%s%s%s%s", FIMPrefix, prefix, FIMSuffix, suffix, FIMMiddle, actualMiddle)

	return TrainingExample{
		Text:     text,
		Source:   f.FilePath,
		Category: "fim",
	}, nil
}
```

Create `internal/formatter/instruction.go`:

```go
package formatter

import (
	"fmt"

	"github.com/yavosh/gocoder/internal/extractor"
)

func FormatInstruction(f extractor.ExtractedFunc) TrainingExample {
	instruction := f.Doc
	if instruction == "" {
		instruction = fmt.Sprintf("Implement the Go function: %s", f.Signature)
	} else {
		instruction = fmt.Sprintf("%s\n\nSignature: %s", instruction, f.Signature)
	}

	return TrainingExample{
		Instruction: instruction,
		Output:      f.Body,
		Source:      f.FilePath,
		Category:   "instruction",
	}
}

func FormatProseInstruction(section extractor.ProseSection) []TrainingExample {
	var examples []TrainingExample

	for _, code := range section.CodeBlocks {
		// Prose -> Code direction
		examples = append(examples, TrainingExample{
			Instruction: section.Prose,
			Output:      code,
			Category:    "prose_to_code",
		})

		// Code -> Explanation direction
		examples = append(examples, TrainingExample{
			Instruction: fmt.Sprintf("Explain this Go code:\n\n%s", code),
			Output:      section.Prose,
			Category:    "code_to_prose",
		})
	}

	return examples
}
```

- [ ] **Step 5: Run tests**

```bash
go test ./internal/formatter/...
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add internal/formatter/
git commit -m "feat: add FIM and instruction formatters"
```

---

## Task 8: Pipeline Build Command (Wire Everything Together)

**Files:**
- Modify: `cmd/gocoder/main.go`
- Create: `internal/pipeline/build.go`
- Create: `internal/pipeline/build_test.go`

- [ ] **Step 1: Write integration test for pipeline build**

Create `internal/pipeline/build_test.go`:

```go
package pipeline_test

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/yavosh/gocoder/internal/pipeline"
)

func TestBuildFromTestdata(t *testing.T) {
	// Create minimal sources.yaml pointing at local test data
	dir := t.TempDir()
	outputDir := t.TempDir()

	// Create a fake Go repo
	repoDir := filepath.Join(dir, "testrepo")
	if err := os.MkdirAll(repoDir, 0o755); err != nil {
		t.Fatal(err)
	}
	goFile := `package main

import "fmt"

// Greet returns a greeting for the given name.
func Greet(name string) string {
	return fmt.Sprintf("Hello, %s!", name)
}

// Add returns the sum of two integers.
func Add(a, b int) int {
	return a + b
}
`
	if err := os.WriteFile(filepath.Join(repoDir, "main.go"), []byte(goFile), 0o644); err != nil {
		t.Fatal(err)
	}

	result, err := pipeline.BuildFromDir(context.Background(), repoDir, outputDir, pipeline.BuildOptions{
		MinLines: 2,
		FIMRatio: 0.5,
		Seed:     42,
	})
	if err != nil {
		t.Fatalf("BuildFromDir: %v", err)
	}

	if result.TotalExamples == 0 {
		t.Error("expected at least one training example")
	}

	// Check output files exist
	if _, err := os.Stat(filepath.Join(outputDir, "train.jsonl")); err != nil {
		t.Error("expected train.jsonl to exist")
	}
	if _, err := os.Stat(filepath.Join(outputDir, "eval.jsonl")); err != nil {
		t.Error("expected eval.jsonl to exist")
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
go test ./internal/pipeline/...
```

Expected: FAIL.

- [ ] **Step 3: Implement pipeline build**

Create `internal/pipeline/build.go`:

```go
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
	MinLines   int
	FIMRatio   float64
	Seed       int64
	EvalSplit  float64 // default 0.1
}

type BuildResult struct {
	TotalExamples  int
	TrainExamples  int
	EvalExamples   int
	FIMExamples    int
	InstrExamples  int
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

	// Shuffle deterministically
	rng.Shuffle(len(examples), func(i, j int) {
		examples[i], examples[j] = examples[j], examples[i]
	})

	// Split
	evalCount := int(float64(len(examples)) * opts.EvalSplit)
	if evalCount < 1 && len(examples) > 0 {
		evalCount = 1
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
```

- [ ] **Step 4: Run tests**

```bash
go test ./internal/pipeline/...
```

Expected: PASS.

- [ ] **Step 5: Add full pipeline Build function that parses sources.yaml and calls collectors**

Create `internal/pipeline/pipeline.go` — the top-level orchestrator that:
1. Parses `sources.yaml` via `config.ParseSources`
2. Calls `collector.CloneAll` for repos
3. Calls `collector.FetchArticle` for each article source
4. Runs `BuildFromDir` on each cloned repo directory
5. Runs `extractor.ExtractProse` + `formatter.FormatProseInstruction` on article content
6. Merges all examples, applies weights from config, writes final JSONL

```go
package pipeline

import (
	"context"
	"fmt"
	"log"

	"github.com/yavosh/gocoder/internal/collector"
	"github.com/yavosh/gocoder/internal/config"
	"github.com/yavosh/gocoder/internal/extractor"
	"github.com/yavosh/gocoder/internal/formatter"
)

func Build(ctx context.Context, sourcesPath, outputDir string, opts BuildOptions) (*BuildResult, error) {
	cfg, err := config.ParseSources(sourcesPath)
	if err != nil {
		return nil, fmt.Errorf("parsing sources: %w", err)
	}

	workDir, err := os.MkdirTemp("", "gocoder-collect-*")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(workDir)

	var allExamples []formatter.TrainingExample

	// Collect and process repos
	for _, repo := range cfg.Repos {
		log.Printf("cloning %s...", repo.URL)
		dir, err := collector.CloneRepo(ctx, repo, workDir)
		if err != nil {
			log.Printf("warning: skipping %s: %v", repo.URL, err)
			continue
		}
		result, err := BuildFromDir(ctx, dir, "", opts) // don't write yet
		if err != nil {
			log.Printf("warning: skipping %s: %v", repo.URL, err)
			continue
		}
		// Apply weight by repeating examples
		weight := repo.EffectiveWeight()
		examples := result.Examples
		if weight > 1.0 {
			for i := 1.0; i < weight; i += 1.0 {
				examples = append(examples, result.Examples...)
			}
		}
		allExamples = append(allExamples, examples...)
	}

	// Collect and process articles
	for _, article := range cfg.Articles {
		log.Printf("fetching article %s...", article.URL)
		content, err := collector.FetchArticle(ctx, article)
		if err != nil {
			log.Printf("warning: skipping article %s: %v", article.URL, err)
			continue
		}
		sections, err := extractor.ExtractProse(content)
		if err != nil {
			continue
		}
		for _, s := range sections {
			allExamples = append(allExamples, formatter.FormatProseInstruction(s)...)
		}
	}

	// Write final JSONL (uses same split/shuffle logic)
	return writeFinalDataset(allExamples, outputDir, opts)
}
```

Note: `BuildFromDir` needs to be refactored to return examples instead of always writing files. Add an `Examples` field to `BuildResult` and a separate `writeFinalDataset` helper.

- [ ] **Step 6: Wire pipeline build into CLI main.go**

Update `cmd/gocoder/main.go` to call `pipeline.Build` when `gocoder pipeline build` is invoked. Accept flags: `--sources` (path to sources.yaml), `--output` (output dir), `--min-lines`, `--fim-ratio`, `--seed`.

- [ ] **Step 6: Verify end-to-end**

```bash
make build && bin/gocoder pipeline build --help
```

Expected: prints usage for pipeline build command.

- [ ] **Step 7: Commit**

```bash
git add internal/pipeline/ cmd/gocoder/main.go
git commit -m "feat: wire pipeline build command end-to-end"
```

---

## Task 9: Eval Runner & Automated Scorer

**Files:**
- Create: `internal/eval/runner.go`
- Create: `internal/eval/scorer.go`
- Create: `internal/eval/types.go`
- Create: `internal/eval/runner_test.go`
- Create: `internal/eval/scorer_test.go`
- Create: `eval/prompts/error_handling.yaml`

- [ ] **Step 1: Define eval types**

Create `internal/eval/types.go`:

```go
package eval

type Prompt struct {
	ID          string `yaml:"id"`
	Category    string `yaml:"category"`
	Description string `yaml:"description"`
	Prompt      string `yaml:"prompt"`
	Weight      float64 `yaml:"weight"`
}

type PromptFile struct {
	Category string   `yaml:"category"`
	Weight   float64  `yaml:"weight"`
	Prompts  []Prompt `yaml:"prompts"`
}

type Result struct {
	PromptID   string  `json:"prompt_id"`
	Category   string  `json:"category"`
	Response   string  `json:"response"`
	Compiles   bool    `json:"compiles"`
	VetClean   bool    `json:"vet_clean"`
	StaticOK   bool    `json:"static_ok"`
	TestsPass  bool    `json:"tests_pass"`
	Score      float64 `json:"score"`
}

type RunOutput struct {
	Model   string   `json:"model"`
	Results []Result `json:"results"`
	Summary Summary  `json:"summary"`
}

type Summary struct {
	TotalPrompts int     `json:"total_prompts"`
	AvgScore     float64 `json:"avg_score"`
	CompileRate  float64 `json:"compile_rate"`
	VetRate      float64 `json:"vet_rate"`
}
```

- [ ] **Step 2: Write test for automated scorer**

Create `internal/eval/scorer_test.go`:

```go
package eval_test

import (
	"testing"

	"github.com/yavosh/gocoder/internal/eval"
)

func TestScoreCompiles(t *testing.T) {
	good := `package main

import "fmt"

func Hello() string {
	return fmt.Sprintf("hello")
}
`
	bad := `package main

func Hello() string {
	return fmt.Sprintf("hello"  // missing paren
}
`

	if !eval.Compiles(good) {
		t.Error("expected good code to compile")
	}
	if eval.Compiles(bad) {
		t.Error("expected bad code to not compile")
	}
}

func TestScoreVet(t *testing.T) {
	clean := `package main

import "fmt"

func Hello() {
	fmt.Println("hello")
}
`
	if !eval.VetClean(clean) {
		t.Error("expected clean code to pass vet")
	}
}
```

- [ ] **Step 3: Run test to verify it fails**

```bash
go test ./internal/eval/...
```

Expected: FAIL.

- [ ] **Step 4: Implement scorer**

Create `internal/eval/scorer.go`:

```go
package eval

import (
	"os"
	"os/exec"
	"path/filepath"
)

// prepareGoEnv creates a temp dir with the source and go.mod, returns dir and cleanup func.
func prepareGoEnv(goSource string) (string, func(), error) {
	dir, err := os.MkdirTemp("", "gocoder-eval-*")
	if err != nil {
		return "", nil, err
	}
	cleanup := func() { os.RemoveAll(dir) }

	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte(goSource), 0o644); err != nil {
		cleanup()
		return "", nil, err
	}

	cmd := exec.Command("go", "mod", "init", "evalcheck")
	cmd.Dir = dir
	if err := cmd.Run(); err != nil {
		cleanup()
		return "", nil, err
	}

	return dir, cleanup, nil
}

func Compiles(goSource string) bool {
	dir, cleanup, err := prepareGoEnv(goSource)
	if err != nil {
		return false
	}
	defer cleanup()

	cmd := exec.Command("go", "build", "./...")
	cmd.Dir = dir
	return cmd.Run() == nil
}

func VetClean(goSource string) bool {
	dir, cleanup, err := prepareGoEnv(goSource)
	if err != nil {
		return false
	}
	defer cleanup()

	cmd := exec.Command("go", "vet", "./...")
	cmd.Dir = dir
	return cmd.Run() == nil
}

// ScoreResult runs all checks in a single temp environment.
func ScoreResult(goSource string) Result {
	dir, cleanup, err := prepareGoEnv(goSource)
	if err != nil {
		return Result{}
	}
	defer cleanup()

	r := Result{}

	buildCmd := exec.Command("go", "build", "./...")
	buildCmd.Dir = dir
	r.Compiles = buildCmd.Run() == nil

	vetCmd := exec.Command("go", "vet", "./...")
	vetCmd.Dir = dir
	r.VetClean = vetCmd.Run() == nil

	score := 0.0
	if r.Compiles {
		score += 0.4
	}
	if r.VetClean {
		score += 0.3
	}
	r.Score = score
	return r
}
```

- [ ] **Step 5: Write a sample prompt file**

Create `eval/prompts/error_handling.yaml`:

```yaml
category: error_handling
weight: 0.15
prompts:
  - id: err_wrap_basic
    description: "Wrap an error with context using fmt.Errorf"
    prompt: "Write a Go function that reads a config file and returns an error wrapped with context if the file cannot be read or parsed."
  - id: err_sentinel
    description: "Define and use sentinel errors"
    prompt: "Write Go code that defines a sentinel error ErrNotFound and a function FindUser that returns it when a user is not found. Include a caller that checks for the error using errors.Is."
  - id: err_as
    description: "Use errors.As for typed error checking"
    prompt: "Write a Go custom error type ValidationError with a Field string. Write a function Validate that returns it, and a caller that extracts the field name using errors.As."
```

- [ ] **Step 6: Implement eval runner**

Create `internal/eval/runner.go` — sends prompts to an OpenAI-compatible API endpoint and collects responses:

```go
package eval

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"
)

type RunConfig struct {
	Endpoint  string // e.g., http://localhost:11434/v1
	Model     string
	OutputDir string
}

func LoadPrompts(promptsDir string) ([]Prompt, error) {
	files, err := filepath.Glob(filepath.Join(promptsDir, "*.yaml"))
	if err != nil {
		return nil, err
	}

	var all []Prompt
	for _, f := range files {
		data, err := os.ReadFile(f)
		if err != nil {
			return nil, fmt.Errorf("reading %s: %w", f, err)
		}

		var pf PromptFile
		if err := yaml.Unmarshal(data, &pf); err != nil {
			return nil, fmt.Errorf("parsing %s: %w", f, err)
		}

		for i := range pf.Prompts {
			pf.Prompts[i].Category = pf.Category
			if pf.Prompts[i].Weight == 0 {
				pf.Prompts[i].Weight = pf.Weight
			}
		}
		all = append(all, pf.Prompts...)
	}
	return all, nil
}

type chatRequest struct {
	Model    string        `json:"model"`
	Messages []chatMessage `json:"messages"`
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatResponse struct {
	Choices []struct {
		Message chatMessage `json:"message"`
	} `json:"choices"`
}

func RunPrompt(ctx context.Context, endpoint, model, prompt string) (string, error) {
	reqBody := chatRequest{
		Model: model,
		Messages: []chatMessage{
			{Role: "system", Content: "You are an expert Go developer. Respond with Go code only, no explanations."},
			{Role: "user", Content: prompt},
		},
	}

	data, err := json.Marshal(reqBody)
	if err != nil {
		return "", err
	}

	url := endpoint + "/chat/completions"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(data))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("HTTP %d: %s", resp.StatusCode, body)
	}

	var chatResp chatResponse
	if err := json.Unmarshal(body, &chatResp); err != nil {
		return "", fmt.Errorf("parsing response: %w", err)
	}

	if len(chatResp.Choices) == 0 {
		return "", fmt.Errorf("no choices in response")
	}

	return chatResp.Choices[0].Message.Content, nil
}
```

- [ ] **Step 7: Run tests**

```bash
go test ./internal/eval/...
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add internal/eval/ eval/prompts/
git commit -m "feat: add eval runner and automated scorer"
```

---

## Task 10: Eval Compare & CLI Wiring

**Files:**
- Create: `internal/eval/compare.go`
- Create: `internal/eval/compare_test.go`
- Modify: `cmd/gocoder/main.go`

- [ ] **Step 1: Write test for comparison**

Create `internal/eval/compare_test.go`:

```go
package eval_test

import (
	"testing"

	"github.com/yavosh/gocoder/internal/eval"
)

func TestCompare(t *testing.T) {
	baseline := &eval.RunOutput{
		Results: []eval.Result{
			{PromptID: "err_wrap", Score: 0.4},
			{PromptID: "err_sentinel", Score: 0.7},
		},
		Summary: eval.Summary{AvgScore: 0.55},
	}

	candidate := &eval.RunOutput{
		Results: []eval.Result{
			{PromptID: "err_wrap", Score: 0.7},
			{PromptID: "err_sentinel", Score: 0.7},
		},
		Summary: eval.Summary{AvgScore: 0.70},
	}

	diff := eval.Compare(baseline, candidate)
	if diff.ScoreDelta <= 0 {
		t.Errorf("expected positive score delta, got %f", diff.ScoreDelta)
	}
	if len(diff.Improved) != 1 {
		t.Errorf("expected 1 improved prompt, got %d", len(diff.Improved))
	}
}
```

- [ ] **Step 2: Implement compare**

Create `internal/eval/compare.go`:

```go
package eval

type ComparisonResult struct {
	ScoreDelta float64
	Improved   []string
	Regressed  []string
	Unchanged  []string
}

func Compare(baseline, candidate *RunOutput) ComparisonResult {
	baseScores := make(map[string]float64)
	for _, r := range baseline.Results {
		baseScores[r.PromptID] = r.Score
	}

	var result ComparisonResult
	result.ScoreDelta = candidate.Summary.AvgScore - baseline.Summary.AvgScore

	for _, r := range candidate.Results {
		base, ok := baseScores[r.PromptID]
		if !ok {
			continue
		}
		switch {
		case r.Score > base+0.05:
			result.Improved = append(result.Improved, r.PromptID)
		case r.Score < base-0.05:
			result.Regressed = append(result.Regressed, r.PromptID)
		default:
			result.Unchanged = append(result.Unchanged, r.PromptID)
		}
	}
	return result
}
```

- [ ] **Step 3: Run tests**

```bash
go test ./internal/eval/...
```

Expected: PASS.

- [ ] **Step 4: Wire eval subcommands into CLI main.go**

Update `cmd/gocoder/main.go` to handle `eval run`, `eval compare`, `eval judge` subcommands with appropriate flags:

- `eval run --endpoint <url> --model <name> --prompts <dir> --output <path>`
- `eval compare --baseline <path> --candidate <path>`
- `eval judge --input <path> --judge <model>`

- [ ] **Step 5: Verify**

```bash
make build && bin/gocoder eval run --help
```

Expected: prints usage for eval run.

- [ ] **Step 6: Commit**

```bash
git add internal/eval/compare.go internal/eval/compare_test.go cmd/gocoder/main.go
git commit -m "feat: add eval compare and wire eval CLI subcommands"
```

---

## Task 11: Model Router Proxy

**Files:**
- Create: `internal/router/config.go`
- Create: `internal/router/proxy.go`
- Create: `internal/router/metrics.go`
- Create: `internal/router/proxy_test.go`
- Create: `serving/config.yaml`

- [ ] **Step 1: Write test for router config parsing and request routing**

Create `internal/router/proxy_test.go`:

```go
package router_test

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/yavosh/gocoder/internal/router"
)

func TestRouteByModel(t *testing.T) {
	cfg := &router.Config{
		Models: map[string]router.ModelConfig{
			"codegen": {Model: "go-nemotron"},
			"fast":    {Model: "qwen3:30b-a3b"},
		},
		Routes: []router.Route{
			{Match: router.Match{Model: "go-*"}, Target: "codegen"},
			{Match: router.Match{Default: true}, Target: "codegen"},
		},
	}

	resolved := cfg.ResolveModel("go-nemotron", "/v1/chat/completions")
	if resolved != "go-nemotron" {
		t.Errorf("expected go-nemotron, got %s", resolved)
	}

	resolved = cfg.ResolveModel("unknown", "/v1/chat/completions")
	if resolved != "go-nemotron" {
		t.Errorf("expected default go-nemotron, got %s", resolved)
	}
}

func TestRouteByPath(t *testing.T) {
	cfg := &router.Config{
		Models: map[string]router.ModelConfig{
			"codegen":      {Model: "go-nemotron"},
			"autocomplete": {Model: "qwen3:30b-a3b"},
		},
		Routes: []router.Route{
			{Match: router.Match{Path: "/v1/completions"}, Target: "autocomplete"},
			{Match: router.Match{Default: true}, Target: "codegen"},
		},
	}

	// Path match should override default
	resolved := cfg.ResolveModel("anything", "/v1/completions")
	if resolved != "qwen3:30b-a3b" {
		t.Errorf("expected qwen3:30b-a3b for /v1/completions, got %s", resolved)
	}

	// Non-matching path falls through to default
	resolved = cfg.ResolveModel("anything", "/v1/chat/completions")
	if resolved != "go-nemotron" {
		t.Errorf("expected go-nemotron for chat, got %s", resolved)
	}
}

func TestProxyForwardsRequest(t *testing.T) {
	// Mock ollama backend
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		if !strings.Contains(string(body), "go-nemotron") {
			t.Errorf("expected model go-nemotron in request body, got %s", body)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"choices":[{"message":{"content":"hello"}}]}`))
	}))
	defer backend.Close()

	cfg := &router.Config{
		Upstream: backend.URL,
		Models: map[string]router.ModelConfig{
			"codegen": {Model: "go-nemotron"},
		},
		Routes: []router.Route{
			{Match: router.Match{Default: true}, Target: "codegen"},
		},
	}

	proxy := router.NewProxy(cfg)
	srv := httptest.NewServer(proxy)
	defer srv.Close()

	reqBody := `{"model":"test","messages":[{"role":"user","content":"hi"}]}`
	resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(reqBody))
	if err != nil {
		t.Fatalf("proxy request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("expected 200, got %d", resp.StatusCode)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
go test ./internal/router/...
```

Expected: FAIL.

- [ ] **Step 3: Implement router config**

Create `internal/router/config.go`:

```go
package router

import (
	"fmt"
	"os"
	"strings"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Listen   string                 `yaml:"listen"`
	Upstream string                 `yaml:"ollama"`
	Models   map[string]ModelConfig `yaml:"models"`
	Routes   []Route                `yaml:"routing"`
}

type ModelConfig struct {
	Model string `yaml:"model"`
}

type Route struct {
	Match  Match  `yaml:"match"`
	Target string `yaml:"model"`
}

type Match struct {
	Path    string `yaml:"path,omitempty"`
	Model   string `yaml:"model,omitempty"`
	Default bool   `yaml:"default,omitempty"`
}

func LoadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading config: %w", err)
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parsing config: %w", err)
	}

	return &cfg, nil
}

func (c *Config) ResolveModel(requestModel string, requestPath string) string {
	for _, route := range c.Routes {
		// Path-based routing (e.g., /v1/completions -> autocomplete)
		if route.Match.Path != "" {
			if strings.HasPrefix(requestPath, route.Match.Path) {
				return c.Models[route.Target].Model
			}
			continue
		}
		// Model-name routing
		if route.Match.Model != "" {
			pattern := route.Match.Model
			if strings.HasSuffix(pattern, "*") {
				prefix := strings.TrimSuffix(pattern, "*")
				if strings.HasPrefix(requestModel, prefix) {
					return c.Models[route.Target].Model
				}
			} else if requestModel == pattern {
				return c.Models[route.Target].Model
			}
			continue
		}
		// Default fallback
		if route.Match.Default {
			return c.Models[route.Target].Model
		}
	}
	return requestModel
}
```

- [ ] **Step 4: Implement proxy**

Create `internal/router/proxy.go`:

```go
package router

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httputil"
	"net/url"
)

func NewProxy(cfg *Config) http.Handler {
	upstream, _ := url.Parse(cfg.Upstream)
	metrics := NewMetrics()

	proxy := &httputil.ReverseProxy{
		Director: func(req *http.Request) {
			req.URL.Scheme = upstream.Scheme
			req.URL.Host = upstream.Host
			req.Host = upstream.Host
		},
	}

	mux := http.NewServeMux()

	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"status":"ok"}`))
	})

	mux.Handle("/metrics", metrics.Handler())

	mux.HandleFunc("/v1/", func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		requestPath := r.URL.Path

		// Read body to extract model
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "failed to read body", http.StatusBadRequest)
			return
		}

		// Parse and rewrite model
		resolvedModel := ""
		var reqBody map[string]any
		if err := json.Unmarshal(body, &reqBody); err == nil {
			if model, ok := reqBody["model"].(string); ok {
				resolvedModel = cfg.ResolveModel(model, requestPath)
				reqBody["model"] = resolvedModel
				rewritten, _ := json.Marshal(reqBody)
				body = rewritten
			}
		}

		r.Body = io.NopCloser(bytes.NewReader(body))
		r.ContentLength = int64(len(body))

		proxy.ServeHTTP(w, r)
		metrics.Record(resolvedModel, time.Since(start), false)
	})

	return mux
}
```

- [ ] **Step 5: Implement basic metrics**

Create `internal/router/metrics.go`:

```go
package router

import (
	"fmt"
	"net/http"
	"sync"
	"time"
)

type Metrics struct {
	mu       sync.Mutex
	requests map[string]int64
	errors   map[string]int64
	latency  map[string][]time.Duration
}

func NewMetrics() *Metrics {
	return &Metrics{
		requests: make(map[string]int64),
		errors:   make(map[string]int64),
		latency:  make(map[string][]time.Duration),
	}
}

func (m *Metrics) Record(model string, duration time.Duration, err bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.requests[model]++
	if err {
		m.errors[model]++
	}
	m.latency[model] = append(m.latency[model], duration)
}

func (m *Metrics) Handler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		m.mu.Lock()
		defer m.mu.Unlock()
		w.Header().Set("Content-Type", "text/plain")
		for model, count := range m.requests {
			fmt.Fprintf(w, "gocoder_requests_total{model=%q} %d\n", model, count)
		}
		for model, count := range m.errors {
			fmt.Fprintf(w, "gocoder_errors_total{model=%q} %d\n", model, count)
		}
	})
}
```

- [ ] **Step 6: Run tests**

```bash
go test ./internal/router/...
```

Expected: PASS.

- [ ] **Step 7: Create serving config**

Create `serving/config.yaml` from the spec.

- [ ] **Step 8: Wire serve command into CLI**

Update `cmd/gocoder/main.go` to handle `gocoder serve --config <path>`.

- [ ] **Step 9: Verify**

```bash
make build && bin/gocoder serve --help
```

Expected: prints usage.

- [ ] **Step 10: Commit**

```bash
git add internal/router/ serving/ cmd/gocoder/main.go
git commit -m "feat: add model routing proxy with OpenAI-compatible API"
```

---

## Task 12: Training Scripts (Python)

**Files:**
- Create: `training/requirements.txt`
- Create: `training/config/nemotron-cascade-2.yaml`
- Create: `training/train.py`
- Create: `training/merge_lora.py`
- Create: `training/convert_gguf.py`

- [ ] **Step 1: Create requirements.txt**

```
unsloth[cu124]>=2024.12
trl>=0.12
peft>=0.13
transformers>=4.46
datasets>=3.0
pyyaml>=6.0
```

- [ ] **Step 2: Create training config YAML**

Create `training/config/nemotron-cascade-2.yaml` from the spec (including the MoE target_modules note).

- [ ] **Step 3: Implement train.py**

Create `training/train.py` — loads config, loads model via unsloth, configures LoRA, runs SFTTrainer on the JSONL dataset. Accept `--config` flag.

Key points:
- Print `model.named_modules()` at startup so user can verify LoRA targets
- Log VRAM usage after model load
- Save checkpoints per `save_steps`
- Log eval loss every `logging_steps`

- [ ] **Step 4: Implement merge_lora.py**

Create `training/merge_lora.py` — merges LoRA adapter weights back into the base model. Accept `--base` and `--adapter` flags.

- [ ] **Step 5: Implement convert_gguf.py**

Create `training/convert_gguf.py` — wraps llama.cpp's conversion and quantization. Accept `--input`, `--output`, `--quant` flags. Checks that llama.cpp tools are available.

- [ ] **Step 6: Verify scripts parse correctly**

```bash
python3 -c "import ast; ast.parse(open('training/train.py').read())"
python3 -c "import ast; ast.parse(open('training/merge_lora.py').read())"
python3 -c "import ast; ast.parse(open('training/convert_gguf.py').read())"
```

Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add training/
git commit -m "feat: add training scripts (train, merge, convert)"
```

---

## Task 13: Ollama Modelfile & Serving Config

**Files:**
- Create: `serving/models/go-nemotron.Modelfile`
- Verify: `serving/config.yaml`

- [ ] **Step 1: Create Modelfile**

Create `serving/models/go-nemotron.Modelfile` from the spec.

- [ ] **Step 2: Verify serving config exists and is valid**

```bash
python3 -c "import yaml; yaml.safe_load(open('serving/config.yaml'))"
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add serving/models/
git commit -m "feat: add ollama Modelfile and serving config"
```

---

## Task 14: Remaining Eval Prompt Files

**Files:**
- Create: `eval/prompts/concurrency.yaml`
- Create: `eval/prompts/http_middleware.yaml`
- Create: `eval/prompts/interfaces.yaml`
- Create: `eval/prompts/testing.yaml`
- Create: `eval/prompts/context_propagation.yaml`
- Create: `eval/prompts/package_design.yaml`
- Create: `eval/prompts/struct_methods.yaml`
- Create: `eval/prompts/idiom.yaml`

- [ ] **Step 1: Create all prompt files**

Each file follows the same structure as `error_handling.yaml`. Write 3-5 prompts per category covering the key patterns described in the spec's category descriptions.

- [ ] **Step 2: Verify all YAML files parse**

```bash
for f in eval/prompts/*.yaml; do python3 -c "import yaml; yaml.safe_load(open('$f'))"; done
```

Expected: no errors.

- [ ] **Step 3: Verify prompt count**

```bash
grep -c "  - id:" eval/prompts/*.yaml | tail -1
```

Expected: at least 27 prompts total (3 per 9 categories).

- [ ] **Step 4: Commit**

```bash
git add eval/prompts/
git commit -m "feat: add eval prompt files for all 9 categories"
```

---

## Task 15: End-to-End Smoke Test & Final Wiring

**Files:**
- Modify: `cmd/gocoder/main.go` (final cleanup)
- Modify: `Makefile`

- [ ] **Step 1: Build and verify all subcommands**

```bash
make build
bin/gocoder pipeline build --help
bin/gocoder eval run --help
bin/gocoder eval compare --help
bin/gocoder serve --help
```

Expected: all print usage without errors.

- [ ] **Step 2: Run full test suite**

```bash
make test
```

Expected: all tests pass.

- [ ] **Step 3: Run lint**

```bash
make lint
```

Expected: no issues.

- [ ] **Step 4: Run pipeline on a small local test repo**

Create a small Go repo with 5-10 functions, run `gocoder pipeline build` against it, verify JSONL output.

```bash
bin/gocoder pipeline build --sources data/sources-test.yaml --output /tmp/gocoder-test
wc -l /tmp/gocoder-test/train.jsonl /tmp/gocoder-test/eval.jsonl
```

Expected: non-zero line counts in both files.

- [ ] **Step 5: Commit any final fixes**

```bash
git add -A && git commit -m "chore: final cleanup and end-to-end verification"
```

---

## Task 16: LLM-as-Judge Implementation

**Files:**
- Create: `internal/eval/judge.go`
- Create: `internal/eval/judge_test.go`

- [ ] **Step 1: Write test for judge**

Create `internal/eval/judge_test.go`:

```go
package eval_test

import (
	"testing"

	"github.com/yavosh/gocoder/internal/eval"
)

func TestFormatJudgePrompt(t *testing.T) {
	prompt := "Write a Go function that wraps errors"
	response := "func wrap(err error) error { return fmt.Errorf(\"op: %w\", err) }"

	judgePrompt := eval.FormatJudgePrompt(prompt, response)
	if judgePrompt == "" {
		t.Error("expected non-empty judge prompt")
	}
}

func TestParseJudgeScore(t *testing.T) {
	response := `{"correctness": 4, "idiom": 5, "simplicity": 4, "completeness": 3}`
	score, err := eval.ParseJudgeScore(response)
	if err != nil {
		t.Fatalf("ParseJudgeScore: %v", err)
	}
	if score.Correctness != 4 {
		t.Errorf("expected correctness 4, got %d", score.Correctness)
	}
	if score.Average() < 3.0 || score.Average() > 5.0 {
		t.Errorf("unexpected average: %f", score.Average())
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
go test ./internal/eval/... -run TestFormatJudge
```

Expected: FAIL.

- [ ] **Step 3: Implement judge**

Create `internal/eval/judge.go`:

```go
package eval

import (
	"context"
	"encoding/json"
	"fmt"
)

type JudgeScore struct {
	Correctness  int `json:"correctness"`
	Idiom        int `json:"idiom"`
	Simplicity   int `json:"simplicity"`
	Completeness int `json:"completeness"`
}

func (s JudgeScore) Average() float64 {
	return float64(s.Correctness+s.Idiom+s.Simplicity+s.Completeness) / 4.0
}

func FormatJudgePrompt(originalPrompt, modelResponse string) string {
	return fmt.Sprintf(`Rate this Go code on a scale of 1-5 for each criterion.
Respond with JSON only: {"correctness": N, "idiom": N, "simplicity": N, "completeness": N}

Original prompt: %s

Code to evaluate:
%s`, originalPrompt, modelResponse)
}

func ParseJudgeScore(response string) (JudgeScore, error) {
	var score JudgeScore
	if err := json.Unmarshal([]byte(response), &score); err != nil {
		return score, fmt.Errorf("parsing judge score: %w", err)
	}
	return score, nil
}

func JudgeResult(ctx context.Context, endpoint, judgeModel, prompt, response string) (JudgeScore, error) {
	judgePrompt := FormatJudgePrompt(prompt, response)
	judgeResponse, err := RunPrompt(ctx, endpoint, judgeModel, judgePrompt)
	if err != nil {
		return JudgeScore{}, fmt.Errorf("running judge: %w", err)
	}
	return ParseJudgeScore(judgeResponse)
}
```

- [ ] **Step 4: Run tests**

```bash
go test ./internal/eval/... -run TestFormatJudge -run TestParseJudge
```

Expected: PASS.

- [ ] **Step 5: Wire eval judge into CLI**

Update `cmd/gocoder/main.go` to handle `gocoder eval judge --input <path> --judge <model> --endpoint <url>`.

- [ ] **Step 6: Commit**

```bash
git add internal/eval/judge.go internal/eval/judge_test.go cmd/gocoder/main.go
git commit -m "feat: add LLM-as-judge eval scoring"
```

---

## Task 17: Blog Index, Sitemap & Markdown Repo Article Collectors

**Files:**
- Modify: `internal/collector/article.go`
- Create: `internal/collector/article_crawl_test.go`

- [ ] **Step 1: Write tests for blog_index and markdown_repo types**

Create `internal/collector/article_crawl_test.go`:

```go
package collector_test

import (
	"context"
	"testing"

	"github.com/yavosh/gocoder/internal/collector"
	"github.com/yavosh/gocoder/internal/config"
)

func TestFetchBlogIndex(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test")
	}

	src := config.ArticleSource{
		URL:  "https://go.dev/blog/",
		Type: "blog_index",
	}

	urls, err := collector.DiscoverArticleURLs(context.Background(), src)
	if err != nil {
		t.Fatalf("DiscoverArticleURLs: %v", err)
	}
	if len(urls) < 10 {
		t.Errorf("expected at least 10 blog URLs, got %d", len(urls))
	}
}

func TestFetchMarkdownRepo(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test")
	}

	src := config.ArticleSource{
		URL:  "https://github.com/uber-go/guide",
		Type: "markdown_repo",
	}

	contents, err := collector.FetchMarkdownRepo(context.Background(), src)
	if err != nil {
		t.Fatalf("FetchMarkdownRepo: %v", err)
	}
	if len(contents) == 0 {
		t.Error("expected at least one markdown file")
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
go test ./internal/collector/... -short
```

Expected: FAIL — functions not defined.

- [ ] **Step 3: Implement blog_index, sitemap, and markdown_repo collectors**

Add to `internal/collector/article.go`:

- `DiscoverArticleURLs` — fetches blog index page, finds `<a>` links to articles, returns list of URLs
- `FetchBlogIndex` — calls DiscoverArticleURLs then fetches each article
- `FetchMarkdownRepo` — clones the repo, finds all `.md` files, returns their content
- `FetchSitemap` — fetches sitemap XML, extracts URLs, fetches each page

Update `FetchArticle` switch to handle all types.

- [ ] **Step 4: Run tests**

```bash
go test ./internal/collector/... -run TestFetchBlogIndex -count=1
```

Expected: PASS (requires network).

- [ ] **Step 5: Commit**

```bash
git add internal/collector/
git commit -m "feat: add blog_index, sitemap, and markdown_repo article collectors"
```

---

## Task 18: FIM Token Verification Step

**Files:**
- Create: `training/verify_fim_tokens.py`

- [ ] **Step 1: Create FIM token verification script**

Create `training/verify_fim_tokens.py`:

```python
"""Verify FIM special tokens for Nemotron-Cascade-2.

Run this BEFORE formatting the dataset. It prints the model's actual
FIM tokens so you can update formatter/fim.go constants if needed.

Usage:
    python training/verify_fim_tokens.py --model nvidia/Nemotron-Cascade-2-30B-A3B
"""
import argparse
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print("=== Special Tokens ===")
    print(f"All special tokens: {tokenizer.all_special_tokens}")
    print(f"Additional special tokens: {tokenizer.additional_special_tokens}")

    # Look for FIM tokens
    fim_candidates = ["fim_prefix", "fim_suffix", "fim_middle",
                      "fill", "prefix", "suffix", "middle"]
    found = {}
    for token in tokenizer.all_special_tokens + tokenizer.additional_special_tokens:
        lower = token.lower()
        for candidate in fim_candidates:
            if candidate in lower:
                found[candidate] = token

    if found:
        print("\n=== FIM Tokens Found ===")
        for name, token in found.items():
            token_id = tokenizer.convert_tokens_to_ids(token)
            print(f"  {name}: {token!r} (id={token_id})")
    else:
        print("\n!!! No FIM tokens found in vocabulary !!!")
        print("You will need to add FIM tokens and resize the embedding layer.")
        print("Add this to train.py before training:")
        print('  tokenizer.add_special_tokens({"additional_special_tokens": ')
        print('    ["<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"]})')
        print('  model.resize_token_embeddings(len(tokenizer))')

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script parses**

```bash
python3 -c "import ast; ast.parse(open('training/verify_fim_tokens.py').read())"
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add training/verify_fim_tokens.py
git commit -m "feat: add FIM token verification script for base model"
```

**Note:** Run this script on the cloud GPU before running `gocoder pipeline build` with FIM formatting. Update `internal/formatter/fim.go` constants if the model uses different tokens.

---

## Task 19: Python Training Script Tests

**Files:**
- Create: `training/test_config.py`

- [ ] **Step 1: Create config validation test**

Create `training/test_config.py`:

```python
"""Basic tests for training config loading and validation."""
import yaml
import os

def test_config_loads():
    config_path = os.path.join(os.path.dirname(__file__), "config", "nemotron-cascade-2.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    assert "model" in cfg, "config must have 'model' section"
    assert "lora" in cfg, "config must have 'lora' section"
    assert "training" in cfg, "config must have 'training' section"
    assert "dataset" in cfg, "config must have 'dataset' section"

    assert cfg["model"]["name"] == "nvidia/Nemotron-Cascade-2-30B-A3B"
    assert cfg["model"]["load_in_4bit"] == False
    assert cfg["training"]["bf16"] == True
    assert cfg["lora"]["r"] in (16, 32)
    assert 0 < cfg["dataset"]["fim_ratio"] < 1

    print("All config tests passed.")

if __name__ == "__main__":
    test_config_loads()
```

- [ ] **Step 2: Run test**

```bash
python3 training/test_config.py
```

Expected: "All config tests passed."

- [ ] **Step 3: Commit**

```bash
git add training/test_config.py
git commit -m "test: add training config validation test"
```

---

## Summary

| Task | Component | Dependency |
|---|---|---|
| 1 | Project scaffold & CLI | None |
| 2 | Sources config parser | Task 1 |
| 3 | Repo collector | Task 2 |
| 4 | Go code extractor | Task 1 |
| 5 | Article collector & prose extractor | Task 2 |
| 6 | Filters | Task 4 |
| 7 | FIM & instruction formatters | Task 4, 6 |
| 8 | Pipeline build command | Task 2, 3, 4, 5, 6, 7 |
| 9 | Eval runner & scorer | Task 1 |
| 10 | Eval compare & CLI wiring | Task 9 |
| 11 | Model router proxy | Task 1 |
| 12 | Training scripts (Python) | None (independent) |
| 13 | Modelfile & serving config | None |
| 14 | Remaining eval prompts | Task 9 |
| 15 | End-to-end smoke test | All except 16-19 |
| 16 | LLM-as-judge | Task 9 |
| 17 | Blog index, sitemap, markdown_repo collectors | Task 5 |
| 18 | FIM token verification | Task 12 (needs model access) |
| 19 | Python training config tests | Task 12 |

**Parallelizable:** Tasks 3+4+5 can run in parallel. Tasks 9+11+12+13 can run in parallel after Task 1. Tasks 14+16 can run in parallel after Task 9. Tasks 17+18+19 can run in parallel.

**Note on issue/PR pairs (B3):** The spec mentions instruction pairs from issue descriptions + fix diffs. This is deferred to a future iteration — it requires GitHub API integration and diff parsing that would significantly expand the pipeline. The core dataset (code + articles) is sufficient for training run 1-7. Add issue/PR pairs as a data expansion step in runs 8-10.
