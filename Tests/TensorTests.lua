require('./BasicTensor.lua')

-- Test for MatMul
mat1 = { { 1, 2, 3 }, { 4, 5, 6 } }
mat2 = { { 1, 2 }, { 3, 4 }, { 5, 6 } }
erg = MatMul( mat1, mat2 )
for i = 1, #erg do
    for j = 1, #erg[1] do
        io.write( erg[i][j] )
        io.write("  ")
    end
    io.write("\n")
end
