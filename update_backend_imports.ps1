# Script to update all imports in backend modules (non-API files)

$backendPath = "backend"
$excludeFolders = @("api", "__pycache__")

$files = Get-ChildItem -Path $backendPath -Filter "*.py" -Recurse | Where-Object {
    $path = $_.FullName
    $exclude = $false
    foreach ($folder in $excludeFolders) {
        if ($path -like "*\$folder\*") {
            $exclude = $true
            break
        }
    }
    -not $exclude
}

$replacements = @(
    @{ old = "from security\."; new = "from backend.security." },
    @{ old = "from database\."; new = "from backend.database." },
    @{ old = "from models\."; new = "from backend.models." },
    @{ old = "from automation\."; new = "from backend.automation." },
    @{ old = "from analytics\."; new = "from backend.analytics." },
    @{ old = "from compliance\."; new = "from backend.compliance." },
    @{ old = "from monitoring\."; new = "from backend.monitoring." },
    @{ old = "from real_time_pipeline\."; new = "from backend.real_time_pipeline." },
    @{ old = "from scalability\."; new = "from backend.scalability." },
    @{ old = "from advanced_ml\."; new = "from backend.advanced_ml." },
    @{ old = "from ai_sales_automation\."; new = "from backend.ai_sales_automation." },
    @{ old = "from customer_experience\."; new = "from backend.customer_experience." },
    @{ old = "from strategic_growth\."; new = "from backend.strategic_growth." },
    @{ old = "import security\."; new = "import backend.security." },
    @{ old = "import database\."; new = "import backend.database." },
    @{ old = "import models\."; new = "import backend.models." },
    @{ old = "import automation\."; new = "import backend.automation." },
    @{ old = "import analytics\."; new = "import backend.analytics." },
    @{ old = "import compliance\."; new = "import backend.compliance." },
    @{ old = "import monitoring\."; new = "import backend.monitoring." },
    @{ old = "import real_time_pipeline\."; new = "import backend.real_time_pipeline." },
    @{ old = "import scalability\."; new = "import backend.scalability." },
    @{ old = "import advanced_ml\."; new = "import backend.advanced_ml." },
    @{ old = "import ai_sales_automation\."; new = "import backend.ai_sales_automation." },
    @{ old = "import customer_experience\."; new = "import backend.customer_experience." },
    @{ old = "import strategic_growth\."; new = "import backend.strategic_growth." }
)

$updatedCount = 0
foreach ($file in $files) {
    $filePath = $file.FullName
    $content = Get-Content $filePath -Raw
    $originalContent = $content
    
    foreach ($replacement in $replacements) {
        $content = $content -replace $replacement.old, $replacement.new
    }
    
    if ($content -ne $originalContent) {
        Set-Content -Path $filePath -Value $content
        Write-Host "Updated: $($file.Name)"
        $updatedCount++
    }
}

Write-Host "Updated $updatedCount files in backend modules!"

