#!/bin/bash
# user_data.sh.tpl
# -----------------
# Runs automatically on first boot of the EC2 instance. Installs Python,
# clones the repo, installs dependencies, and starts the FastAPI service
# via systemd so it survives reboots and restarts on failure.
#
# NOTE: replace the git clone URL below with your own repo once it's
# public on GitHub.

set -e

dnf update -y
dnf install -y python3.11 python3.11-pip git

REPO_DIR=/opt/enterprise-rag-search
git clone https://github.com/YOUR_USERNAME/enterprise-rag-search.git $REPO_DIR
cd $REPO_DIR

python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Inject the API key as an environment file, loaded by python-dotenv
cat > $REPO_DIR/.env << EOF
GROQ_API_KEY=${groq_api_key}
EOF

# systemd service so the API restarts automatically and survives reboots
cat > /etc/systemd/system/rag-api.service << EOF
[Unit]
Description=Enterprise RAG Search API
After=network.target

[Service]
WorkingDirectory=$REPO_DIR/src
ExecStart=$REPO_DIR/venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000
Restart=always
User=ec2-user

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable rag-api
systemctl start rag-api
