package collector

import (
	"context"
	"fmt"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/yavosh/gocoder/internal/config"
)

// ResolveRepo returns the local directory for a repo source.
// For local dirs, it returns the path directly. For URLs, it clones to outputDir.
func ResolveRepo(ctx context.Context, src config.RepoSource, outputDir string) (string, error) {
	if src.IsLocal() {
		dir := src.Dir
		if src.Path != "" {
			dir = filepath.Join(dir, src.Path)
		}
		return dir, nil
	}

	return CloneRepo(ctx, src, outputDir)
}

func CloneRepo(ctx context.Context, src config.RepoSource, outputDir string) (string, error) {
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
