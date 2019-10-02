import System.Environment
import Data.Char
import Data.List.Split
import Data.List

main :: IO(String)
main = do
  arg <- getArgs
  list <- readFile (head arg)
  let list1 = [x|x <- list, x /= '}' && x /= '{']
  let list2 = [if x == ',' then ' ' else x|x <- list1]
  writeFile ("_"++[x|x<-(head arg),(not (isLower x))]++".txt") list2 
  return list2
