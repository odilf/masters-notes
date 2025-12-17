build-all:
    mkdir -p "build" && \
    cd content && \
    for subject in `ls .`; do \
        if [[ $subject != *.typ ]] then \
            typst compile "$subject/notes.typ" "../build/$subject.pdf" --root .; \
        fi \
    done
    cd content && typst compile "analysis/exercises.typ" "../build/analysis-exercises.pdf" --root .;
    cd content && typst compile "differential-equations/exercises.typ" "../build/differential-equations-exercises.pdf" --root .;
