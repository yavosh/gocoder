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
	return fmt.Sprintf("hello"
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

func TestScoreResult(t *testing.T) {
	good := `package main

import "fmt"

func Hello() {
	fmt.Println("hello")
}
`
	r := eval.ScoreResult(good)
	if !r.Compiles {
		t.Error("expected compiles = true")
	}
	if !r.VetClean {
		t.Error("expected vet_clean = true")
	}
	if r.Score < 0.5 {
		t.Errorf("expected score >= 0.5, got %f", r.Score)
	}
}
