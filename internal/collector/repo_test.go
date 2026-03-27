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
