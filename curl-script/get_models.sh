set -x
port=${1:-"8001"}
curl http://localhost:$port/v1/models -H "Content-Type: application/json" -H "Authorization: Bearer intel123"
