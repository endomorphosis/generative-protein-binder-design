# Project File Organization

This document describes the file organization structure for the generative-protein-binder-design project.

## Directory Structure

```
generative-protein-binder-design/
├── README.md                    # Main project documentation
├── LICENSE                       # Apache 2.0 license
├── CODE_OF_CONDUCT.md           # Community guidelines
├── SECURITY.md                   # Security policies
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore patterns
├── docs/                         # All documentation files
│   ├── ARM64_*.md                # ARM64-specific guides (18 files)
│   ├── LOCAL_SETUP.md            # Local development setup
│   ├── SYSTEM_VERIFICATION.md    # System compatibility check
│   ├── GITHUB_ACTIONS_COMPLETE.md
│   ├── RUNNER_TEST.md
│   ├── WORKFLOW_TRIGGER_TEST.md
│   ├── FILE_ORGANIZATION.md      # This file
│   └── Protein_Design_Architecture_Diagram.png
├── scripts/                      # All shell scripts
│   ├── detect_platform.sh        # Platform detection
│   ├── setup_local.sh            # Local setup automation
│   ├── setup_github_runner.sh    # GitHub runner setup
│   ├── check_runner_status.sh    # Runner status check
│   ├── verify_arm64_port.sh      # ARM64 verification
│   ├── continue_arm64_port.sh    # ARM64 porting helper
│   ├── trigger_*.sh              # Workflow triggers
│   ├── print_env.sh              # Environment info
│   └── [ARM64 installation scripts]
├── tests/                        # Test files
│   ├── README.md                 # Test documentation
│   ├── test_attention_solutions.py
│   └── test_flash_attention_working.py
├── src/                          # Source code
│   ├── protein-binder-design.ipynb
│   └── arm64_cuda_fallback/
├── deploy/                       # Deployment configurations
├── tools/                        # External tools
├── .github/                      # GitHub workflows and templates
└── [Helm charts and other directories]
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
