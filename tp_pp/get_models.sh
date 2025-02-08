set +x
port=$1
curl http://localhost:$port/v1/models -H "Content-Type: application/json" -H "Authorization: Bearer intel123"
