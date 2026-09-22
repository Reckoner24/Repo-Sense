from tree_sitter import Language

Language.build_library(
    'build/my-languages.so',
    [
        'tree-sitter-scala',  # Ruta a la gramática descargada
        # Puedes agregar más lenguajes aquí
    ]
)