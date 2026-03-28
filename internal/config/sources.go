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
	Dir    string  `yaml:"dir"`
	Path   string  `yaml:"path"`
	Weight float64 `yaml:"weight"`
}

func (r RepoSource) IsLocal() bool {
	return r.Dir != ""
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
