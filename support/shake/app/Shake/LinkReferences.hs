{-# LANGUAGE OverloadedStrings #-}
-- | Link Agda identifiers in inline code blocks.
module Shake.LinkReferences (linkReferences) where

import qualified Data.HashMap.Strict as HashMap
import Data.HashMap.Strict (HashMap)
import Data.Maybe
import qualified Data.Text as T
import Data.Text (Text)

import Text.Pandoc.Definition
import Text.HTML.TagSoup
import Text.Pandoc.Walk

-- | Link inline code spans of the form `foo`{.Agda} to `foo` if it exists in
-- the module's Agda blocks. If an identifier is explicitly specified with an
-- attribute but doesn't exist, raise an error.
linkReferences :: FilePath -> Pandoc -> Pandoc
linkReferences modname (Pandoc meta blocks) = Pandoc meta (walk link blocks)
  where
    hm :: HashMap Text Reference
    hm = parseSymbolRefs blocks

    link :: Inline -> Inline
    link inline@(Code (_, classes, kv) text)
      | isToBeLinked =
        case HashMap.lookup identifier hm of
          Just ref -> renderReference ref text
          Nothing | isJust ident -> error $ "Could not find identifier " ++ T.unpack identifier ++ " for the inline reference in " ++ modname
                  | otherwise -> inline
      where
        classes' = map T.toLower classes

        isToBeLinked = ("agda" `elem` classes')
                    && ("nolink" `notElem` classes')

        ident = lookup "ident" kv

        identifier = case ident of
          Just id -> id
          _ -> text
    link x = x

renderReference :: Reference -> Text -> Inline
renderReference (Reference href cls ty) t =
  Span ("", ["Agda"], []) [Link ("", [cls], maybeToList (("data-identifier",) <$> ty)) [Code nullAttr t] (href, "")]

data Reference =
  Reference { refHref  :: Text
            , refClass :: Text
            , refType  :: Maybe Text
            }
  deriving (Eq, Show)

-- | Find all links in Agda code blocks (represented as HTML not
-- markdown) and build a map of ident -> reference.
parseSymbolRefs :: [Block] -> HashMap Text Reference
parseSymbolRefs = go mempty . concatMap getHTML where
  getHTML :: Block -> [Tag Text]
  getHTML (RawBlock "html" xs) = parseTags xs
  getHTML (BlockQuote bs) = bs >>= getHTML
  getHTML (Div _ bs) = bs >>= getHTML
  getHTML _ = []

  go :: HashMap Text Reference -> [Tag Text] -> HashMap Text Reference
  go map (TagOpen "a" meta:TagText t:TagClose "a":xs)
    | Just cls <- lookup "class" meta
    , Just href <- lookup "href" meta
    = go (addIfNotPresent t (Reference href cls (lookup "data-identifier" meta)) map) xs

    | otherwise = go map xs

  go map (_:xs) = go map xs
  go map [] = map

addIfNotPresent :: Text -> v -> HashMap Text v -> HashMap Text v
addIfNotPresent = HashMap.insertWith (\_ old -> old)
