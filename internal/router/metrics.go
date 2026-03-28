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

func (m *Metrics) Record(model string, duration time.Duration, isErr bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.requests[model]++
	if isErr {
		m.errors[model]++
	}
	m.latency[model] = append(m.latency[model], duration)
}

func (m *Metrics) Handler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		m.mu.Lock()
		defer m.mu.Unlock()
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		for model, count := range m.requests {
			fmt.Fprintf(w, "# TYPE gocoder_requests_total counter\n")
			fmt.Fprintf(w, "gocoder_requests_total{model=%q} %d\n", model, count)
		}
		for model, count := range m.errors {
			fmt.Fprintf(w, "# TYPE gocoder_errors_total counter\n")
			fmt.Fprintf(w, "gocoder_errors_total{model=%q} %d\n", model, count)
		}
	})
}
