#CNNの行列積への変形

##CNN

input =
{
  {
    {input111,input112,input113,input114},
    {input121,input122,input123,input124},
    {input131,input132,input133,input134},
    {input141,input142,input143,input144}
  },
    {input211,input212,input213,input214},
    {input221,input222,input223,input224},
    {input231,input232,input233,input234},
    {input241,input242,input243,input244}
  }
}

W =
{
  {
    {
      {W1111,W1112,W1113},
      {W1121,W1122,W1123},
      {W1131,W1132,W1133},
    },
    {
      {W1211,W1212,W1213},
      {W1221,W1222,W1223},
      {W1231,W1232,W1233},
    }
  },
  {
    {
      {W2111,W2112,W2113},
      {W2121,W2122,W2123},
      {W2131,W2132,W2133},
    },
    {
      {W2211,W2212,W2213},
      {W2221,W2222,W2223},
      {W2231,W2232,W2233},
    }
  }
}

b = {b1,b1}

output =
{
  {
    {input111 * W1111 + input112 * W1112 + input113 * W1113 + input121 * W1121 + input122 * W1122 + input123 * W1123 + input131 * W1131 + input132 * W1132 + input133 * W1133 +

     input211 * W1211 + input212 * W1212 + input213 * W1213 + input221 * W1221 + input222 * W1222 + input223 * W1223 + input231 * W1231 + input232 * W1232 + input233 * W1233 +

     b1,

     input112 * W1111 + input113 * W1112 + input114 * W1113 + input122 * W1121 + input123 * W1122 + input124 * W1123 + input132 * W1131 + input133 * W1132 + input134 * W1133 +

     input212 * W1211 + input213 * W1212 + input214 * W1213 + input222 * W1221 + input223 * W1222 + input224 * W1223 + input232 * W1231 + input233 * W1232 + input234 * W1233 +

     b1
    },
    {input121 * W1111 + input122 * W1112 + input123 * W1113 + input131 * W1121 + input132 * W1122 + input133 * W1123 + input141 * W1131 + input142 * W1132 + input143 * W1133 +

     input221 * W1211 + input222 * W1212 + input223 * W1213 + input231 * W1221 + input232 * W1222 + input233 * W1223 + input241 * W1231 + input242 * W1232 + input243 * W1233 +

     b1,

     input122 * W1111 + input123 * W1112 + input124 * W1113 + input132 * W1121 + input133 * W1122 + input134 * W1123 + input142 * W1131 + input143 * W1132 + input144 * W1133 +

     input222 * W1211 + input223 * W1212 + input224 * W1213 + input232 * W1221 + input233 * W1222 + input234 * W1223 + input242 * W1231 + input243 * W1232 + input244 * W1233 +

     b1
    }
  },
  {
    {input111 * W2111 + input112 * W2112 + input113 * W2113 + input121 * W2121 + input122 * W2122 + input123 * W2123 + input131 * W2131 + input132 * W2132 + input133 * W2133 +

     input211 * W2211 + input212 * W2212 + input213 * W2213 + input221 * W2221 + input222 * W2222 + input223 * W2223 + input231 * W2231 + input232 * W2232 + input233 * W2233 +

     b2,

     input112 * W2111 + input113 * W2112 + input114 * W2113 + input122 * W2121 + input123 * W2122 + input124 * W2123 + input132 * W2131 + input133 * W2132 + input134 * W2133 +

     input212 * W2211 + input213 * W2212 + input214 * W2213 + input222 * W2221 + input223 * W2222 + input224 * W2223 + input232 * W2231 + input233 * W2232 + input234 * W2233 +

     b2
    },
    {input121 * W2111 + input122 * W2112 + input123 * W2113 + input131 * W2121 + input132 * W2122 + input133 * W2123 + input141 * W2131 + input142 * W2132 + input143 * W2133 +

     input221 * W2211 + input222 * W2212 + input223 * W2213 + input231 * W2221 + input232 * W2222 + input233 * W2223 + input241 * W2231 + input242 * W2232 + input243 * W2233 +

     b2,

     input122 * W2111 + input123 * W2112 + input124 * W2113 + input132 * W2121 + input133 * W2122 + input134 * W2123 + input142 * W2131 + input143 * W2132 + input144 * W2133 +

     input222 * W2211 + input223 * W2212 + input224 * W2213 + input232 * W2221 + input233 * W2222 + input234 * W2223 + input242 * W2231 + input243 * W2232 + input244 * W2233 +

     b2
    }
  }
}

##行列積

input =
{
  {input111, input112, input121, input122},
  {input112, input113, input122, input123},
  {input113, input114, input123, input124},

  {input121, input122, input131, input132},
  {input122, input123, input132, input133},
  {input123, input124, input133, input134},

  {input131, input132, input141, input142},
  {input132, input133, input142, input143},
  {input133, input134, input143, input144},

  {input211, input212, input221, input222},
  {input212, input213, input222, input223},
  {input213, input214, input223, input224},

  {input221, input222, input231, input232},
  {input222, input223, input232, input233},
  {input223, input224, input233, input234},

  {input231, input232, input241, input242},
  {input232, input233, input242, input243},
  {input233, input234, input243, input244}
}

W =
{
  {W1111,W1112,W1113, W1121,W1122,W1123, W1131,W1132,W1133, W1211,W1212,W1213, W1221,W1222,W1223, W1231,W1232,W1233},
  {W2111,W2112,W2113, W2121,W2122,W2123, W2131,W2132,W2133, W2211,W2212,W2213, W2221,W2222,W2223, W2231,W2232,W2233},
}

b =
{
  {b1},
  {b2}
}

W * input + b