{-# LANGUAGE BlockArguments, OverloadedStrings, RankNTypes, ScopedTypeVariables #-}

-- | Build a graph of links between nodes.
module Shake.LinkGraph
  ( linksRules
  , getInternalLinks
  ) where

import Control.Monad
import Control.Monad.IO.Class

import qualified Data.Map as Map
import qualified Data.Set as Set
import qualified Data.Text as Text
import qualified Data.Text.IO as Text
import Data.Aeson
import Data.Char (isDigit)
import Data.List
import Data.Maybe
import Data.Text (Text)

import Development.Shake
import Development.Shake.FilePath

import HTML.Base (Identifier(Identifier))

import Network.URI

import Shake.Modules
import Shake.Options
import Shake.SearchData
import Shake.Utils

import Text.HTML.TagSoup

-- | These modules should not appear in the dependency graph.
ignoreLinks :: Set.Set String
ignoreLinks = Set.fromList [ "all-pages", "index" ]

linksRules :: Rules ()
linksRules = do
  -- A set of valid link targets.
  anchors <- newCache \() -> do
    allModules <- getAllModules
    let moduleAnchors = Set.fromList [ Text.pack (mod <.> "html") | mod <- Map.keys allModules ]
    searchData :: [SearchTerm] <- readJSONFile "_build/html/static/search.json"
    let searchAnchors = Set.fromList (map idAnchor searchData)
    agdaIdents :: [Identifier] <- readJSONFile "_build/all-types.json"
    let agdaAnchors = Set.fromList [ Text.concat [filename, "#", ident]
                                   | Identifier ident anchor _type _tooltip <- agdaIdents
                                   , let (filename, _) = Text.break (== '#') anchor ]
    pure $ Set.unions [moduleAnchors, searchAnchors, agdaAnchors]

  -- This file contains a list of [source, target] links representing a directed
  -- graph of module names *with no self-loops*.
  -- While building this file, we also check links against the set of anchors above.
  "_build/html/static/links.json" %> \out -> do
    skipTypes <- getSkipTypes
    skipAgda <- getSkipAgda

    ourModules <- getOurModules
    anchors <- if skipAgda then pure mempty else anchors ()
    links :: [[String]] <- catMaybes . concat <$> forM (Map.keys ourModules) \source -> do
      let input = "_build/html" </> source <.> "html"
      need [input]
      links <- Set.toList . getInternalLinks source . parseTags <$> liftIO (Text.readFile input)
      forM links \link -> do
        unless (skipTypes || Text.pack link `Set.member` anchors) do
          error $ "Could not find link target " ++ link ++ " in " ++ source
        let target = dropExtension . fst $ break (== '#') link
        pure if (  target /= source
                && target `Map.member` ourModules
                && source `Set.notMember` ignoreLinks
                && target `Set.notMember` ignoreLinks)
             then Just [source, target]
             else Nothing
    liftIO $ encodeFile out links

-- | Extract internal links to 1Lab modules, with anchors. Ignore numeric
-- links because they're always correct and make the graph too crowded.
getInternalLinks :: String -> [Tag Text] -> Set.Set String
getInternalLinks mod = foldMap go where
  go (TagOpen "a" attrs)
    | Just href <- lookup "href" attrs
    , Just uri <- parseRelativeReference (Text.unpack href)
    = maybe mempty (\target -> Set.singleton (target ++ uriFragment uri)) $
        case pathSegments uri of
          [target] | '#':anchor@(_:_) <- uriFragment uri, all isDigit anchor -> Nothing
                   | otherwise -> Just target
          [] | "/" `isPrefixOf` uriPath uri -> Just "index.html"
             | not ("#fn" `isPrefixOf` uriFragment uri)
             , not ("#cb" `isPrefixOf` uriFragment uri)
             , not ("#ref-" `isPrefixOf` uriFragment uri) -> Just (mod <.> "html")
          _ -> Nothing
  go _ = mempty
