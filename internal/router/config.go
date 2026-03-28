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
		if route.Match.Path != "" {
			if strings.HasPrefix(requestPath, route.Match.Path) {
				return c.Models[route.Target].Model
			}
			continue
		}
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
		if route.Match.Default {
			return c.Models[route.Target].Model
		}
	}
	return requestModel
}
