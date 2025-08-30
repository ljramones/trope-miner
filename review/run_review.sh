# (optional) set DB path explicitly
export TROPES_DB="$(pwd)/../ingester/tropes.db"
echo "Tropes DB is at:" + $TROPES_DB

export FLASK_DEBUG=1
export FLASK_ENV=development

# install Flask (in your venv)
pip install flask matplotlib

# start the review UI (dev server on http://127.0.0.1:5050)
python app.py


