{-# LANGUAGE OverloadedStrings, TypeFamilies #-}
{-# LANGUAGE DeriveGeneric, DeriveAnyClass, DerivingStrategies #-}
{-# LANGUAGE ViewPatterns, BlockArguments, LambdaCase #-}
module Definitions
  ( glossaryRules
  , WikiLink(..)
  , isWikiLink
  , getWikiLink, getWikiLinkUrl
  , Glossary(getEntries), GlossaryQ(..)
  , Mangled(getMangled), mangleLink
  , Definition(..), definitionTarget
  )
  where

import Agda.Utils.Impossible

import Control.Monad.IO.Class
import Control.DeepSeq
import Control.Arrow (first)

import qualified Data.Map.Strict as Map
import qualified Data.Text.IO as Text
import qualified Data.Text as Text
import Data.Map.Strict (Map)
import Data.Function
import Data.Hashable
import Data.Foldable
import Data.Monoid ( Endo(..) )
import Data.Binary ( Binary )
import Data.List (intersperse, sortOn, groupBy)
import Data.Text (Text)
import Data.Char
import Data.Ord (Down(..))

import Development.Shake.FilePath
import Development.Shake

import GHC.Generics (Generic)

import HTML.Backend (moduleName)

import Text.Pandoc.Walk
import Text.Pandoc

import {-# SOURCE #-} Shake.Markdown (readLabMarkdown)

newtype Mangled = Mangled { getMangled :: Text }
  deriving (Show, Eq, Ord, Generic)
  deriving anyclass (Binary, NFData, Hashable)

mangleLink :: Text -> Mangled
mangleLink = doit where
  doit
    = Mangled
    . Text.concat
    . intersperse (Text.singleton '-')
    . map (Text.filter wordChar . Text.toLower)
    . Text.words

  wordChar '-' = True
  wordChar '[' = True
  wordChar c = isAsciiLower c || isDigit c

parseDefinitions :: MonadIO m => FilePath -> FilePath -> m Glossary
parseDefinitions anchor input = liftIO do
  Pandoc _meta markdown <- readLabMarkdown input
  pure $ appEndo (query (definitionBlock input anchor) markdown) (Glossary mempty)

data Definition = Definition
  { definitionModule :: FilePath
  , definitionAnchor :: Text
  , definitionCopy   :: Bool
  }
  deriving (Show, Generic, NFData, Binary)

instance Eq Definition where
  (==) = (==) `on` definitionAnchor

instance Hashable Definition where
  hashWithSalt s = hashWithSalt s . definitionAnchor

definitionBlock :: FilePath -> FilePath -> Block -> Endo Glossary
definitionBlock inp fp = go where
  mod = dropExtension fp
  add id v = Endo $ addDefinition (mangleLink v) Definition
    { definitionModule = mod
    , definitionAnchor = id
    , definitionCopy   = False
    }

  addMany id = foldMap (add id) . Text.words

  go (Div (id, [only], keys) _blocks) | "definition" == only, not (Text.null id) =
    let aliases = foldMap (addMany id) (lookup "alias" keys)
    in add id id <> aliases

  go (Header _ (id, _, keys) _inline) =
    foldMap (addMany id) (lookup "defines" keys)

  go (CodeBlock (_, [], _) _) = error $ "Code block without class in " ++ inp

  go _ = mempty

newtype Glossary = Glossary { getEntries :: Map Mangled Definition }
  deriving (Show, Eq, Generic)
  deriving anyclass (Binary, NFData, Hashable)

data GlossaryQ       = GlossaryQ deriving (Eq, Show, Generic, NFData, Binary, Hashable)
newtype ModuleGlossaryQ = ModuleGlossaryQ FilePath
  deriving newtype (Eq, Show, NFData, Binary, Hashable)
newtype LinkTargetQ     = LinkTargetQ Text
  deriving newtype (Eq, Show, NFData, Binary, Hashable)

type instance RuleResult GlossaryQ       = Glossary
type instance RuleResult ModuleGlossaryQ = Glossary
type instance RuleResult LinkTargetQ     = (Text, Text)

addDefinition :: Mangled -> Definition -> Glossary -> Glossary
addDefinition _ Definition{definitionCopy = True} ge = ge
addDefinition key@(getMangled -> keyt) def (Glossary ge) = Glossary (go False key (plural ge)) where
  plural
    | Text.null keyt = error $ "Definition has empty key: " ++ show def
    | Text.last keyt == 'y' = go True (Mangled (Text.init keyt <> "ies"))
    | otherwise             = go True (Mangled (keyt <> "s"))
  go c key ge = case Map.lookup key ge of
    Just def' | def' /= def -> error $ unlines
      [ "Conflict when building link map:"
      , "The files " ++ show (definitionModule def) ++ " and " ++ show (definitionModule def')
        ++ " both define the term " ++ show key
      ]
    _ -> Map.insert key def{definitionCopy = c} ge

definitionTarget :: Definition -> Text
definitionTarget def = Text.pack (definitionModule def) <> ".html#" <> definitionAnchor def

glossaryRules :: Rules ()
glossaryRules = do
  _ <- addOracleCache \(ModuleGlossaryQ fp) -> do
    need [fp]
    let modn = moduleName (dropExtensions (dropDirectory1 fp)) <.> "html"
    traced "parsing definitions" (parseDefinitions modn fp)

  _ <- addOracle \GlossaryQ -> do
    md   <- fmap ("src" </>) <$> getDirectoryFiles "src" ["**/*.lagda.md"]
    need md
    outs <- askOracles (ModuleGlossaryQ <$> md)

    let
      alldefs :: [(Mangled, Definition)]
      alldefs = outs >>= (Map.toList . getEntries)

    pure $ foldr (uncurry addDefinition) (Glossary mempty) alldefs

  _ <- addOracle \(LinkTargetQ target) -> do
    glo <- getEntries <$> askOracle GlossaryQ
    case Map.lookup (mangleLink target) glo of
      Just def -> pure (definitionAnchor def, definitionTarget def)
      Nothing  -> error $
        "Unknown wiki-link target: " ++ Text.unpack target

  -- Debugging target to print all the wikilink targets:
  phony "glossary" do
    let
      get = sortOn (Down . length . snd)
          . map (\case { xs@((_, d):_) -> (definitionModule d, xs) ; _ -> __IMPOSSIBLE__ })
          . groupBy ((==) `on` definitionModule . snd)
          . sortOn (definitionModule . snd)
          . map (first getMangled)
          . Map.toList . getEntries

      bykey =
          groupBy ((==) `on` definitionAnchor . snd)
        . sortOn (definitionAnchor . snd)

      aliases [(k, d)] = "  #" <> definitionAnchor d <> ": " <> k
      aliases g@((_, d):_) =
        "  #" <> definitionAnchor d <> ":\n    "
        <> Text.intercalate ", " (map fst g)
      aliases _ = __IMPOSSIBLE__

    entries <- get <$> askOracle GlossaryQ
    liftIO $ for_ entries $ \(mod, defs) -> do
      Text.putStr $ Text.pack (dropExtension mod) <> ":\n"
        <> Text.unlines (aliases <$> bykey defs)

  pure ()

data WikiLink = WikiLink
  { wikiLinkDest     :: Text
  , wikiLinkContents :: [Inline]
  , wikiLinkAttrs    :: Attr
  }
  deriving Show

instance Eq WikiLink where
  (==) = (==) `on` wikiLinkDest

instance Hashable WikiLink where
  hashWithSalt s = hashWithSalt s . wikiLinkDest

isWikiLink :: Inline -> Maybe WikiLink
isWikiLink (Link attr contents (url, title))
  | "wikilink" == title = pure $ WikiLink url contents attr
isWikiLink _ = Nothing

getWikiLinkUrl :: Text -> Action (Text, Text)
getWikiLinkUrl = askOracle . LinkTargetQ

getWikiLink :: WikiLink -> Action Inline
getWikiLink (WikiLink dest contents (id, cls, kv)) = do
  (anchor, url) <- getWikiLinkUrl dest
  pure $ Link (id, cls, ("data-target", anchor):kv) contents (url, "")
