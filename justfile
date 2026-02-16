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

build SUBJECT:
    typst compile content/{{SUBJECT}}/notes.typ build/{{SUBJECT}}.pdf --root content

watch SUBJECT:
    typst watch content/{{SUBJECT}}/notes.typ build/{{SUBJECT}}.pdf --root content

open SUBJECT:
    xdg-open build/{{SUBJECT}}.pdf

wb SUBJECT:
    #!/usr/bin/env -S parallel --shebang --ungroup --jobs {{ num_cpus() }}
    just open {{SUBJECT}}
    just watch {{SUBJECT}}
