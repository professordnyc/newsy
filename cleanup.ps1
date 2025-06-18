# Script to clean up temporary and test files

# Create archive directories if they don't exist
$archiveDirs = @(
    "archive/scripts",
    "archive/test_results",
    "archive/mcp_configs"
)

foreach ($dir in $archiveDirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

# Move test scripts to archive/scripts
$testScripts = @(
    "test_*.py",
    "analyze_*.py",
    "direct_serpapi_test.py",
    "inspect_json.py",
    "extract_serpapi_structure.py",
    "run_newsy.py"
)

foreach ($pattern in $testScripts) {
    Get-ChildItem -Path . -Filter $pattern -File | ForEach-Object {
        Move-Item -Path $_.FullName -Destination "archive/scripts/" -Force
        Write-Host "Moved $($_.Name) to archive/scripts/"
    }
}

# Move test result JSON files to archive/test_results
$testResults = @(
    "*.json"
)

foreach ($pattern in $testResults) {
    Get-ChildItem -Path . -Filter $pattern -File | 
        Where-Object { $_.Name -notmatch "config\.json|mcp\.json" } |
        ForEach-Object {
            Move-Item -Path $_.FullName -Destination "archive/test_results/" -Force
            Write-Host "Moved $($_.Name) to archive/test_results/"
        }
}

# Clean up old MCP configs (keeping only v1)
if (Test-Path "mcp/1") {
    if (Test-Path "mcp/1/src") {
        Get-ChildItem -Path "mcp/1" -Exclude "src" | Remove-Item -Recurse -Force
    }
    Move-Item -Path "mcp/1/*.json" -Destination "archive/mcp_configs/" -Force -ErrorAction SilentlyContinue
}

# Remove any __pycache__ directories
Get-ChildItem -Path . -Directory -Recurse -Filter "__pycache__" | Remove-Item -Recurse -Force

Write-Host "Cleanup complete. Temporary files have been moved to archive/ directory."
