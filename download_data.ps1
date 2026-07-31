$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force data/raw | Out-Null
kaggle competitions download -c jigsaw-agile-community-rules -p data/raw
Expand-Archive -LiteralPath data/raw/jigsaw-agile-community-rules.zip -DestinationPath data/raw -Force
Get-ChildItem data/raw
