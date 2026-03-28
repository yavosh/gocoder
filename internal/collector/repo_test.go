package collector_test

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/yavosh/gocoder/internal/collector"
	"github.com/yavosh/gocoder/internal/config"
)

func TestResolveLocalRepo(t *testing.T) {
	// Create a local dir with a Go file
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "main.go"), []byte("package main\n"), 0o644)

	src := config.RepoSource{
		Dir:    dir,
		Weight: 2.0,
	}

	resolved, err := collector.ResolveRepo(context.Background(), src, t.TempDir())
	if err != nil {
		t.Fatalf("ResolveRepo: %v", err)
	}
	if resolved != dir {
		t.Errorf("expected %s, got %s", dir, resolved)
	}
}

func TestResolveLocalRepoWithPath(t *testing.T) {
	dir := t.TempDir()
	sub := filepath.Join(dir, "sub")
	os.MkdirAll(sub, 0o755)
	os.WriteFile(filepath.Join(sub, "lib.go"), []byte("package lib\n"), 0o644)

	src := config.RepoSource{
		Dir:  dir,
		Path: "sub",
	}

	resolved, err := collector.ResolveRepo(context.Background(), src, t.TempDir())
	if err != nil {
		t.Fatalf("ResolveRepo: %v", err)
	}
	if resolved != sub {
		t.Errorf("expected %s, got %s", sub, resolved)
	}
}

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

	if info, err := os.Stat(repoDir); err != nil || !info.IsDir() {
		t.Errorf("expected repoDir to be a directory at examples/, got error: %v", err)
	}
}
