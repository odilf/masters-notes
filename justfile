build-all:
    mkdir -p "build" && \
    cd content && \
    for subject in `ls .`; do \
        if [[ $subject != *.typ ]] then \
            typst compile "$subject/notes.typ" "../build/$subject.pdf" --root .; \
        fi \
    done
