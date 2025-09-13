build-all:
    for file in `ls content/**/*`; do \
        echo "$file" && \
        mkdir -p "build/$(dirname "$file")" && \
        typst compile "$file/notes.typ" "build/$file.pdf" --root content; \
    done
