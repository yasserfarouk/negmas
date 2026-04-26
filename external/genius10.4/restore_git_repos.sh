
# Repository: projects/negmas/external/genius10.4
echo -e "${BLUE}Restoring: projects/negmas/external/genius10.4${NC}"
if [ -d "projects/negmas/external/genius10.4/.git" ]; then
    echo -e "  ${YELLOW}Directory already exists, skipping...${NC}"
else
    # Create parent directory if needed
    mkdir -p "projects/negmas/external"
    
    # Clone the repository
    if git clone "git@github.com:yasserfarouk/genius_source_code.git" "projects/negmas/external/genius10.4"; then
        echo -e "  ${GREEN}✓${NC} Successfully cloned"
        
        # Checkout the original branch if not already on it
        cd "projects/negmas/external/genius10.4"
        current=$(git rev-parse --abbrev-ref HEAD)
        if [ "$current" != "main" ]; then
            if git checkout "main" 2>/dev/null; then
                echo -e "  ${GREEN}✓${NC} Checked out branch: main"
            else
                echo -e "  ${YELLOW}⚠${NC} Could not checkout branch: main"
            fi
        fi
        cd - > /dev/null
    else
        echo -e "  ${RED}✗${NC} Failed to clone"
    fi
fi


# Repository: projects/negmas/external/genius10.4
echo -e "${BLUE}Restoring: projects/negmas/external/genius10.4${NC}"
if [ -d "projects/negmas/external/genius10.4/.git" ]; then
    echo -e "  ${YELLOW}Directory already exists, skipping...${NC}"
else
    # Create parent directory if needed
    mkdir -p "projects/negmas/external"
    
    # Clone the repository
    if git clone "git@github.com:yasserfarouk/genius_source_code.git" "projects/negmas/external/genius10.4"; then
        echo -e "  ${GREEN}✓${NC} Successfully cloned"
        
        # Checkout the original branch if not already on it
        cd "projects/negmas/external/genius10.4"
        current=$(git rev-parse --abbrev-ref HEAD)
        if [ "$current" != "main" ]; then
            if git checkout "main" 2>/dev/null; then
                echo -e "  ${GREEN}✓${NC} Checked out branch: main"
            else
                echo -e "  ${YELLOW}⚠${NC} Could not checkout branch: main"
            fi
        fi
        cd - > /dev/null
    else
        echo -e "  ${RED}✗${NC} Failed to clone"
    fi
fi

