{-# LANGUAGE BlockArguments, GeneralizedNewtypeDeriving, TypeFamilies #-}

-- | Shake rules to compile equations to HTML with KaTeX
module Shake.KaTeX
  ( katexRules
  , getDisplayMath
  , getInlineMath
  , prerenderMaths

  , getParsedPreamble
  , getPreambleFor
  ) where

import Control.Monad

import qualified Data.ByteString.Lazy as LazyBS
import qualified Data.Text.Encoding as Text
import qualified Data.Text.IO as Text
import qualified Data.Text as Text
import Data.Text (Text)
import Data.Generics

import Development.Shake.Classes (Hashable, Binary, NFData)
import Development.Shake

import Shake.Utils
import Macros

newtype LatexEquation = LatexEquation (Bool, Text)
  deriving (Show, Typeable, Eq, Hashable, Binary, NFData)

newtype LatexPreamble = LatexPreamble Bool
  deriving (Show, Typeable, Eq, Hashable, Binary, NFData)

newtype ParsedPreamble = ParsedPreamble ()
  deriving (Show, Typeable, Eq, Hashable, Binary, NFData)

type instance RuleResult LatexEquation = Text
type instance RuleResult LatexPreamble = Text
type instance RuleResult ParsedPreamble = Preamble

-- | Convert a display/block-level LaTeX equation to HTML.
getDisplayMath :: Text -> Action Text
getDisplayMath contents = askOracle (LatexEquation (True, contents))

-- | Convert an inline LaTeX equation to HTML.
getInlineMath :: Text -> Action Text
getInlineMath contents = askOracle (LatexEquation (False, contents))

prerenderMaths :: [Text] -> [Text] -> Action ()
prerenderMaths display inline = void . askOracles $ [LatexEquation (True, c) | c <- display] <> [LatexEquation (False, c) | c <- inline]

-- | Get the preamble for a given build ('True' = dark mode)
getPreambleFor :: Bool -> Action Text
getPreambleFor = askOracle . LatexPreamble

-- | Get parsed preamble
getParsedPreamble :: Action Preamble
getParsedPreamble = askOracle $ ParsedPreamble ()

-- | Shake rules required for compiling KaTeX equations.
katexRules :: Rules ()
katexRules = versioned 2 do
  _ <- addOracle \(ParsedPreamble _) -> do
    need ["src/preamble.tex"]
    t <- liftIO $ Text.readFile "src/preamble.tex"
    case parsePreamble t of
      Just p -> pure p
      Nothing -> error "Failed to parse preamble.tex!"

  _ <- addOracleCache \(LatexPreamble bool) -> do
    pre <- askOracle (ParsedPreamble ())
    let dark = if bool then darkSettings else mempty
    pure (preambleToLatex pre <> Text.pack "\n" <> dark)

  _ <- versioned 2 $ addOracleCache \(LatexEquation (display, tex)) -> do
    pre <- askOracle (ParsedPreamble ())

    let args = ["-T", "-F", "html"] ++ ["-d" | display]
        stdin = LazyBS.fromStrict $ Text.encodeUtf8 $ applyPreamble pre tex

    Stdout out <- nodeCommand [StdinBS stdin] "katex" args
    pure . Text.stripEnd . Text.decodeUtf8 $ out

  pure ()

darkSettings :: Text
darkSettings = Text.pack "\\newcommand{\\labdark}{yes!}"
