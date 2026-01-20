import csv
import random
import numpy as np
import uuid
from datetime import datetime, timedelta

# -----------------------------
# CONFIGURATION
# -----------------------------
NUM_RECORDS = 10000  # number of service metric records (simulate 7 days)
OUTPUT_FILE = "microservices_load_data.csv"

# Microservices in the architecture
SERVICES = {
    "api-gateway": {"base_rps": 1000, "cpu_intensity": 0.3, "memory_mb": 512, "criticality": "critical"},
    "user-service": {"base_rps": 200, "cpu_intensity": 0.5, "memory_mb": 1024, "criticality": "high"},
    "product-catalog": {"base_rps": 400, "cpu_intensity": 0.6, "memory_mb": 2048, "criticality": "high"},
    "order-service": {"base_rps": 150, "cpu_intensity": 0.8, "memory_mb": 1024, "criticality": "critical"},
    "payment-service": {"base_rps": 120, "cpu_intensity": 0.7, "memory_mb": 512, "criticality": "critical"},
    "inventory-service": {"base_rps": 180, "cpu_intensity": 0.6, "memory_mb": 1024, "criticality": "high"},
    "notification-service": {"base_rps": 300, "cpu_intensity": 0.4, "memory_mb": 512, "criticality": "medium"},
    "analytics-service": {"base_rps": 100, "cpu_intensity": 0.9, "memory_mb": 2048, "criticality": "low"},
    "recommendation-engine": {"base_rps": 250, "cpu_intensity": 0.9, "memory_mb": 4096, "criticality": "medium"},
    "review-service": {"base_rps": 80, "cpu_intensity": 0.5, "memory_mb": 1024, "criticality": "low"},
    "cart-service": {"base_rps": 220, "cpu_intensity": 0.6, "memory_mb": 1024, "criticality": "high"},
    "shipping-service": {"base_rps": 90, "cpu_intensity": 0.5, "memory_mb": 512, "criticality": "medium"},
    "search-service": {"base_rps": 350, "cpu_intensity": 0.7, "memory_mb": 2048, "criticality": "high"},
    "image-service": {"base_rps": 500, "cpu_intensity": 0.4, "memory_mb": 1024, "criticality": "medium"},
    "logging-service": {"base_rps": 800, "cpu_intensity": 0.3, "memory_mb": 512, "criticality": "low"},
}

# Service dependencies (caller -> callees)
SERVICE_DEPENDENCIES = {
    "api-gateway": ["user-service", "product-catalog", "order-service", "search-service"],
    "order-service": ["payment-service", "inventory-service", "notification-service"],
    "product-catalog": ["search-service", "recommendation-engine", "image-service"],
    "recommendation-engine": ["product-catalog", "analytics-service"],
    "user-service": ["notification-service", "analytics-service"],
    "payment-service": ["notification-service", "logging-service"],
    "review-service": ["notification-service", "analytics-service"],
    "cart-service": ["product-catalog", "inventory-service"],
}

# Load balancing algorithms
LB_ALGORITHMS = [
    "round-robin",
    "least-connections",
    "weighted-response-time",
    "random",
    "ip-hash",
    "ml-predictive",
]

# Pod health states
HEALTH_STATES = ["healthy", "degraded", "unhealthy"]


# -----------------------------
# UTILITY FUNCTIONS
# -----------------------------
def get_time_multiplier(hour, day_of_week):
    """Traffic patterns by time of day and day of week."""
    # Base pattern by hour
    if 0 <= hour <= 5:
        time_factor = 0.2  # night
    elif 6 <= hour <= 8:
        time_factor = 0.6  # morning
    elif 9 <= hour <= 11:
        time_factor = 1.2  # mid-morning
    elif 12 <= hour <= 13:
        time_factor = 1.5  # lunch peak
    elif 14 <= hour <= 17:
        time_factor = 1.3  # afternoon
    elif 18 <= hour <= 20:
        time_factor = 1.8  # evening peak
    elif 21 <= hour <= 23:
        time_factor = 0.9  # late evening
    else:
        time_factor = 1.0
    
    # Weekend modifier (lower traffic)
    day_factor = 0.7 if day_of_week >= 5 else 1.0
    
    # Random promotional spikes (5% chance)
    spike = random.uniform(2.0, 4.0) if random.random() < 0.05 else 1.0
    
    return time_factor * day_factor * spike


def compute_request_rate(service_name, hour, day_of_week, base_rps):
    """Calculate current request rate for a service."""
    multiplier = get_time_multiplier(hour, day_of_week)
    rate = base_rps * multiplier
    
    # Add noise
    noise = np.random.normal(0, rate * 0.1)
    return max(1, int(rate + noise))


def compute_latency(request_rate, num_pods, cpu_util, lb_algorithm, service_criticality):
    """Calculate service latency based on load and configuration."""
    # Base latency by criticality
    base_latency = {
        "critical": 50,
        "high": 80,
        "medium": 120,
        "low": 200,
    }
    
    base = base_latency.get(service_criticality, 100)
    
    # Load factor (more requests per pod = higher latency)
    rps_per_pod = request_rate / max(1, num_pods)
    load_factor = 1 + (rps_per_pod / 100) * 0.5  # increases with load
    
    # CPU utilization impact
    if cpu_util > 90:
        cpu_factor = 3.0
    elif cpu_util > 80:
        cpu_factor = 2.0
    elif cpu_util > 70:
        cpu_factor = 1.5
    else:
        cpu_factor = 1.0
    
    # Load balancer efficiency
    lb_efficiency = {
        "round-robin": 1.0,
        "least-connections": 0.9,
        "weighted-response-time": 0.85,
        "random": 1.1,
        "ip-hash": 1.05,
        "ml-predictive": 0.75,  # best performance
    }
    
    effective_latency = base * load_factor * cpu_factor * lb_efficiency.get(lb_algorithm, 1.0)
    
    # Add noise
    noise = np.random.normal(0, effective_latency * 0.15)
    
    return max(10, int(effective_latency + noise))


def compute_p95_p99_latency(avg_latency, error_rate):
    """Calculate tail latencies."""
    # P95 is typically 1.5-2x average
    p95_mult = 1.8 if error_rate < 0.05 else 2.5
    p95 = int(avg_latency * p95_mult)
    
    # P99 is typically 2.5-4x average
    p99_mult = 3.0 if error_rate < 0.05 else 5.0
    p99 = int(avg_latency * p99_mult)
    
    return p95, p99


def compute_cpu_utilization(request_rate, num_pods, cpu_intensity):
    """Calculate CPU utilization percentage."""
    # Base CPU per request
    base_cpu_per_request = cpu_intensity * 0.01
    
    # Total CPU needed
    total_cpu = request_rate * base_cpu_per_request
    
    # CPU available (assume 1 core per pod)
    available_cpu = num_pods * 100
    
    cpu_util = (total_cpu / available_cpu) * 100
    
    # Add noise
    noise = np.random.normal(0, 5)
    
    return max(5, min(100, cpu_util + noise))


def compute_memory_utilization(request_rate, num_pods, base_memory_mb):
    """Calculate memory utilization percentage."""
    # Memory per request (connections, cache)
    memory_per_request = 2  # MB
    
    # Active connections (assume 10% of RPS are concurrent)
    active_connections = request_rate * 0.1
    memory_used = base_memory_mb + (active_connections * memory_per_request)
    
    # Available memory
    available_memory = num_pods * base_memory_mb
    
    memory_util = (memory_used / available_memory) * 100
    
    # Add noise
    noise = np.random.normal(0, 5)
    
    return max(10, min(95, memory_util + noise))


def compute_error_rate(cpu_util, memory_util, health_status, num_pods, request_rate):
    """Calculate error rate based on resource pressure."""
    base_error = 0.001  # 0.1% baseline
    
    # CPU pressure
    if cpu_util > 95:
        cpu_error = 0.15
    elif cpu_util > 85:
        cpu_error = 0.05
    elif cpu_util > 75:
        cpu_error = 0.01
    else:
        cpu_error = 0.0
    
    # Memory pressure
    if memory_util > 90:
        mem_error = 0.10
    elif memory_util > 80:
        mem_error = 0.03
    else:
        mem_error = 0.0
    
    # Health status
    health_error = {
        "healthy": 0.0,
        "degraded": 0.05,
        "unhealthy": 0.30,
    }
    
    # Cascading failure risk (high RPS with few pods)
    if num_pods < 3 and request_rate > 200:
        cascade_error = 0.08
    else:
        cascade_error = 0.0
    
    total_error = base_error + cpu_error + mem_error + health_error.get(health_status, 0.0) + cascade_error
    
    return min(0.5, total_error)  # cap at 50%


def determine_health_status(cpu_util, memory_util, error_rate):
    """Determine pod health status."""
    if error_rate > 0.10 or cpu_util > 95 or memory_util > 90:
        return "unhealthy"
    elif error_rate > 0.05 or cpu_util > 85 or memory_util > 80:
        return "degraded"
    else:
        return "healthy"


def should_scale(cpu_util, memory_util, error_rate, queue_depth):
    """Determine if auto-scaling should trigger."""
    scale_up = (cpu_util > 75 or memory_util > 75 or error_rate > 0.05 or queue_depth > 100)
    scale_down = (cpu_util < 30 and memory_util < 40 and error_rate < 0.01 and queue_depth < 10)
    
    if scale_up:
        return "scale-up"
    elif scale_down:
        return "scale-down"
    else:
        return "stable"


# -----------------------------
# FIELD NAMES
# -----------------------------
FIELDNAMES = [
    "record_id",
    "timestamp",
    "hour",
    "day_of_week",
    "service_name",
    "pod_id",
    "num_pods",
    "request_rate_rps",
    "avg_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "cpu_utilization",
    "memory_utilization",
    "error_rate",
    "timeout_rate",
    "active_connections",
    "queue_depth",
    "health_status",
    "lb_algorithm",
    "cache_hit_rate",
    "network_throughput_mbps",
    "retry_count",
    "circuit_breaker_state",
    "scaling_event",
    "dependency_latency_ms",
    "cascading_failure_risk",
    # ML Target variables (to be filled in post-processing)
    "target_request_rate_5min",
    "target_request_rate_10min",
    "target_p99_latency_5min",
    "target_cpu_util_5min",
    "will_overload_5min",
    "will_fail_5min",
    "optimal_num_pods",
    "best_lb_algorithm",
]


# -----------------------------
# DATA GENERATION
# -----------------------------
start_date = datetime(2025, 11, 25, 0, 0, 0)
current_time = start_date

data = []

# Track current pod counts and states
service_states = {}
for service in SERVICES.keys():
    service_states[service] = {
        "num_pods": random.randint(3, 8),
        "last_scale_time": current_time,
    }

for i in range(NUM_RECORDS):
    hour = current_time.hour
    day_of_week = current_time.weekday()
    
    # Select service (weighted by base traffic)
    service_weights = [SERVICES[s]["base_rps"] for s in SERVICES.keys()]
    service_name = random.choices(list(SERVICES.keys()), weights=service_weights, k=1)[0]
    service_info = SERVICES[service_name]
    state = service_states[service_name]
    
    num_pods = state["num_pods"]
    pod_id = f"{service_name}-{random.randint(1, num_pods)}"
    
    # Request rate
    request_rate = compute_request_rate(
        service_name, hour, day_of_week, service_info["base_rps"]
    )
    
    # Load balancing algorithm (can change over time)
    lb_algorithm = random.choices(
        LB_ALGORITHMS,
        weights=[0.25, 0.20, 0.15, 0.10, 0.10, 0.20],  # ml-predictive gets 20%
        k=1
    )[0]
    
    # Resource utilization
    cpu_util = compute_cpu_utilization(request_rate, num_pods, service_info["cpu_intensity"])
    memory_util = compute_memory_utilization(request_rate, num_pods, service_info["memory_mb"])
    
    # Health status
    error_rate = compute_error_rate(cpu_util, memory_util, "healthy", num_pods, request_rate)
    health_status = determine_health_status(cpu_util, memory_util, error_rate)
    
    # Latency
    avg_latency = compute_latency(
        request_rate, num_pods, cpu_util, lb_algorithm, service_info["criticality"]
    )
    p95_latency, p99_latency = compute_p95_p99_latency(avg_latency, error_rate)
    
    # Timeout rate (related to error rate)
    timeout_rate = error_rate * 0.6  # 60% of errors are timeouts
    
    # Active connections (proportional to RPS)
    active_connections = int(request_rate * random.uniform(0.08, 0.15))
    
    # Queue depth (increases with overload)
    if cpu_util > 85:
        queue_depth = int(np.random.exponential(50))
    elif cpu_util > 70:
        queue_depth = int(np.random.exponential(20))
    else:
        queue_depth = int(np.random.exponential(5))
    
    # Cache hit rate
    cache_hit_rate = random.uniform(0.60, 0.95)
    
    # Network throughput (MB/s)
    network_throughput = round(request_rate * random.uniform(0.5, 2.0) / 100, 2)
    
    # Retry count (higher under stress)
    retry_count = int(np.random.poisson(error_rate * 10))
    
    # Circuit breaker state
    if error_rate > 0.20:
        circuit_breaker_state = "open"
    elif error_rate > 0.10:
        circuit_breaker_state = "half-open"
    else:
        circuit_breaker_state = "closed"
    
    # Auto-scaling decision
    scaling_event = should_scale(cpu_util, memory_util, error_rate, queue_depth)
    
    # Apply scaling (with cooldown)
    if (current_time - state["last_scale_time"]).total_seconds() > 300:  # 5 min cooldown
        if scaling_event == "scale-up" and num_pods < 15:
            state["num_pods"] = min(15, num_pods + random.randint(1, 3))
            state["last_scale_time"] = current_time
        elif scaling_event == "scale-down" and num_pods > 2:
            state["num_pods"] = max(2, num_pods - 1)
            state["last_scale_time"] = current_time
        else:
            scaling_event = "stable"
    else:
        scaling_event = "cooldown"
    
    # Dependency latency (if service has dependencies)
    if service_name in SERVICE_DEPENDENCIES:
        num_dependencies = len(SERVICE_DEPENDENCIES[service_name])
        dependency_latency = int(avg_latency * 0.3 * num_dependencies)
    else:
        dependency_latency = 0
    
    # Cascading failure risk score (0-100)
    if service_info["criticality"] == "critical" and error_rate > 0.10:
        cascading_risk = min(100, int(error_rate * 200 + cpu_util * 0.5))
    else:
        cascading_risk = int(error_rate * 100)
    
    # Create row
    row = {
        "record_id": str(uuid.uuid4())[:8],
        "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
        "hour": hour,
        "day_of_week": day_of_week,
        "service_name": service_name,
        "pod_id": pod_id,
        "num_pods": num_pods,
        "request_rate_rps": request_rate,
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": p95_latency,
        "p99_latency_ms": p99_latency,
        "cpu_utilization": round(cpu_util, 2),
        "memory_utilization": round(memory_util, 2),
        "error_rate": round(error_rate, 4),
        "timeout_rate": round(timeout_rate, 4),
        "active_connections": active_connections,
        "queue_depth": queue_depth,
        "health_status": health_status,
        "lb_algorithm": lb_algorithm,
        "cache_hit_rate": round(cache_hit_rate, 2),
        "network_throughput_mbps": network_throughput,
        "retry_count": retry_count,
        "circuit_breaker_state": circuit_breaker_state,
        "scaling_event": scaling_event,
        "dependency_latency_ms": dependency_latency,
        "cascading_failure_risk": cascading_risk,
        # Placeholder for targets (will be filled after all data is generated)
        "target_request_rate_5min": None,
        "target_request_rate_10min": None,
        "target_p99_latency_5min": None,
        "target_cpu_util_5min": None,
        "will_overload_5min": None,
        "will_fail_5min": None,
        "optimal_num_pods": None,
        "best_lb_algorithm": None,
    }
    
    data.append(row)
    
    # Advance time (metrics collected every 30-60 seconds)
    current_time += timedelta(seconds=random.randint(30, 60))

# -----------------------------
# POST-PROCESS: CREATE ML TARGETS
# -----------------------------
print("\nCreating ML target variables...")

# Group data by service for time-series targets
service_data = {}
for idx, row in enumerate(data):
    service = row["service_name"]
    if service not in service_data:
        service_data[service] = []
    service_data[service].append((idx, row))

# Create targets for each service's time series
for service_name, service_records in service_data.items():
    for i, (idx, row) in enumerate(service_records):
        # Find future records (5 min = ~6 records, 10 min = ~12 records at 50sec avg interval)
        future_5min_idx = min(i + 6, len(service_records) - 1)
        future_10min_idx = min(i + 12, len(service_records) - 1)
        
        # Time-series forecasting targets
        future_5min = service_records[future_5min_idx][1]
        future_10min = service_records[future_10min_idx][1]
        
        data[idx]["target_request_rate_5min"] = future_5min["request_rate_rps"]
        data[idx]["target_request_rate_10min"] = future_10min["request_rate_rps"]
        data[idx]["target_p99_latency_5min"] = future_5min["p99_latency_ms"]
        data[idx]["target_cpu_util_5min"] = future_5min["cpu_utilization"]
        
        # Binary classification: will overload in 5 min?
        will_overload = 1 if (future_5min["cpu_utilization"] > 85 or 
                             future_5min["error_rate"] > 0.05 or
                             future_5min["memory_utilization"] > 85) else 0
        data[idx]["will_overload_5min"] = will_overload
        
        # Binary classification: will fail in 5 min?
        will_fail = 1 if (future_5min["error_rate"] > 0.20 or
                         future_5min["health_status"] == "unhealthy") else 0
        data[idx]["will_fail_5min"] = will_fail
        
        # Optimal number of pods (based on future load)
        future_rps = future_5min["request_rate_rps"]
        # Rule of thumb: ~100 RPS per pod for balanced load
        optimal_pods = max(2, min(15, int(np.ceil(future_rps / 100))))
        data[idx]["optimal_num_pods"] = optimal_pods
        
        # Best LB algorithm (find which one gives lowest latency for similar conditions)
        # Look at recent records with similar load
        similar_records = []
        for j in range(max(0, i-20), min(i+20, len(service_records))):
            other_idx, other_row = service_records[j]
            if abs(other_row["request_rate_rps"] - row["request_rate_rps"]) < 50:
                similar_records.append(other_row)
        
        if similar_records:
            # Find algorithm with best average latency in similar conditions
            algo_latencies = {}
            for rec in similar_records:
                algo = rec["lb_algorithm"]
                if algo not in algo_latencies:
                    algo_latencies[algo] = []
                algo_latencies[algo].append(rec["avg_latency_ms"])
            
            best_algo = min(algo_latencies.items(), 
                          key=lambda x: np.mean(x[1]))[0] if algo_latencies else row["lb_algorithm"]
            data[idx]["best_lb_algorithm"] = best_algo
        else:
            data[idx]["best_lb_algorithm"] = row["lb_algorithm"]

print(f"Created target variables for {len(data)} records across {len(service_data)} services")

# Write to CSV
with open(OUTPUT_FILE, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(data)

print(f"\n{'='*70}")
print(f"MICROSERVICES LOAD BALANCING DATA GENERATED")
print(f"{'='*70}")
print(f"Total records: {len(data)}")
print(f"Date range: {data[0]['timestamp']} to {data[-1]['timestamp']}")
print(f"\nServices: {len(SERVICES)}")
for service, info in SERVICES.items():
    print(f"  - {service}: {info['criticality']} priority")

# Summary statistics
total_errors = sum(d['error_rate'] for d in data) / len(data)
avg_latency = sum(d['avg_latency_ms'] for d in data) / len(data)
avg_p99 = sum(d['p99_latency_ms'] for d in data) / len(data)
scale_events = sum(1 for d in data if d['scaling_event'] in ['scale-up', 'scale-down'])
unhealthy_count = sum(1 for d in data if d['health_status'] == 'unhealthy')

print(f"\nSummary Statistics:")
print(f"  Avg error rate: {total_errors*100:.2f}%")
print(f"  Avg latency: {avg_latency:.0f}ms")
print(f"  Avg P99 latency: {avg_p99:.0f}ms")
print(f"  Scaling events: {scale_events}")
print(f"  Unhealthy instances: {unhealthy_count} ({unhealthy_count/len(data)*100:.1f}%)")

print(f"\nLoad Balancing Algorithms:")
for algo in LB_ALGORITHMS:
    count = sum(1 for d in data if d['lb_algorithm'] == algo)
    print(f"  - {algo}: {count} ({count/len(data)*100:.1f}%)")

print(f"\nFile saved: '{OUTPUT_FILE}'")
print(f"{'='*70}\n")
