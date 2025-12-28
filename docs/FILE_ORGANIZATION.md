# Project File Organization

This document describes the file organization structure for the generative-protein-binder-design project.

## Directory Structure

```
generative-protein-binder-design/
├── README.md                    # Main project documentation
├── LICENSE                      # MIT license
├── CODE_OF_CONDUCT.md           # Community guidelines
├── SECURITY.md                  # Security policies
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore patterns
├── .pre-commit-config.yaml      # Pre-commit hooks
├── .workflow-trigger            # CI/CD trigger (ARM64 workflows)
├── docs/                         # All documentation files
│   ├── ARM64_*.md                # ARM64-specific guides
│   ├── ALPHAFOLD_*.md            # AlphaFold optimization docs
│   ├── MMSEQS2_*.md              # MMseqs2 integration docs
│   ├── IMPLEMENTATION_*.md       # Implementation reports
│   ├── TESTING_SUMMARY.txt       # Testing summaries
│   ├── FILE_ORGANIZATION.md      # This file
│   └── images/, demos/           # Assets and example scripts
├── scripts/                      # All shell scripts and installers
│   ├── run_dashboard_stack.sh    # Start/stop Dashboard + MCP stack
│   ├── submit_demo_job.sh        # Submit demo job to MCP
│   ├── doctor_stack.sh           # Health/diagnostics
│   ├── install_all_native.sh     # Zero-touch native installer (AlphaFold + MMseqs2)
│   ├── convert_alphafold_db_to_mmseqs2_multistage.sh # MMseqs2 converter
│   ├── install_mmseqs2.sh        # MMseqs2 binary installer
│   ├── test_mmseqs2_installer_integration.sh # Installer verification
│   └── [other utility scripts]
├── tests/                        # Test files and resources
│   ├── api_requests.http         # API sample calls (renamed from requests.http)
│   ├── test_mcp_tools.py
│   ├── test_mcp_validation.py
│   ├── test_vscode_integration.py
│   ├── test-cicd.sh
│   ├── bench_mmseqs2_lookup.sh
│   └── [other tests]
├── src/                          # Source code
├── deploy/                       # Deployment configurations (compose, Helm)
├── tools/                        # External tools (alphafold2, rfdiffusion, proteinmpnn)
├── mcp-server/                   # MCP server implementation
├── mcp-dashboard/                # Dashboard frontend
├── mcp-dashboard-real/           # Legacy/alt dashboard
├── .github/                      # GitHub workflows and templates
└── user-container/, outputs/, benchmarks/  # Supporting assets
```

## Rationale

### Root Directory
The root directory now contains only essential files that users need to see immediately:
- Project README
- License and conduct documents
- Basic requirements file

### docs/ Directory
All documentation has been moved to `docs/` for:
- Cleaner root directory
- Better organization of ARM64-specific guides
- Easier navigation for users looking for documentation
- Standard practice in open-source projects

### scripts/ Directory
All shell scripts have been moved to `scripts/` for:
- Clear separation of executable scripts from documentation
- Easier script management and maintenance
- Standard practice for utility scripts
- Better organization for CI/CD workflows

### tests/ Directory
Test files have been moved to `tests/` for:
- Standard Python project structure
- Clear separation of tests from source code
- Easier test discovery and execution
- Future expansion of test suite

## Updating References

When referencing these files:

**From root directory:**
```bash
./scripts/detect_platform.sh
./scripts/setup_local.sh
```

**From docs/ directory:**
```bash
../scripts/detect_platform.sh
```

**Documentation links in README:**
```markdown
[Local Setup Guide](docs/LOCAL_SETUP.md)
[ARM64 Guide](docs/ARM64_COMPLETE_GUIDE.md)
```

## Migration Notes

All references in the following locations have been updated:
- ✅ README.md - Updated all script and doc paths
- ✅ .github/workflows/ - Updated workflow script paths
- ✅ .github/ISSUE_TEMPLATE/ - Updated script references
- ✅ docs/*.md - Updated cross-references between docs

No code functionality has been changed, only file locations and references.
