package eval

import (
	"os"
	"os/exec"
	"path/filepath"
)

// prepareGoEnv creates a temp dir with the source and go.mod.
func prepareGoEnv(goSource string) (string, func(), error) {
	dir, err := os.MkdirTemp("", "gocoder-eval-*")
	if err != nil {
		return "", nil, err
	}
	cleanup := func() { os.RemoveAll(dir) }

	if err := os.WriteFile(filepath.Join(dir, "check.go"), []byte(goSource), 0o644); err != nil {
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
