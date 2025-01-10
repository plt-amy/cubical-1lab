{-# LANGUAGE BlockArguments, ScopedTypeVariables #-}
{-# LANGUAGE GeneralizedNewtypeDeriving, TypeFamilies #-}
{-# LANGUAGE OverloadedStrings #-}
module Shake.AgdaCompile (agdaRules) where

import System.FilePath

import Control.Monad.Except

import qualified Data.List.NonEmpty as List1
import qualified Data.HashMap.Strict as Hm
import qualified Agda.Utils.BiMap as BiMap
import qualified Data.Map as Map
import qualified Data.Text as T
import Data.Aeson (eitherDecodeFileStrict', encodeFile)
import Data.Maybe (fromMaybe)
import Data.Foldable
import Data.IORef

import Development.Shake.Classes
import Development.Shake

import Shake.Options (getSkipTypes, getWatching)

import Agda.Compiler.Backend hiding (getEnv)
import Agda.Interaction.FindFile (SourceFile(..))
import Agda.TypeChecking.Monad.Base (srcFromPath)
import Agda.TypeChecking.Pretty.Warning
import Agda.TypeChecking.Errors
import Agda.Interaction.Imports
import Agda.Interaction.Options
import Agda.Syntax.Common (Cubical(CFull))
import Agda.Syntax.Common.Pretty
import Agda.Syntax.TopLevelModuleName
  ( TopLevelModuleName'(..)
  , TopLevelModuleName
  , RawTopLevelModuleName(..)
  , hashRawTopLevelModuleName
  )
import Agda.Syntax.Position (noRange)
import qualified Agda.Utils.Maybe.Strict as S
import qualified Agda.Utils.Trie as Trie
import Agda.Utils.FileName
import Agda.Utils.Hash (Hash)
import Agda.Utils.Lens ((^.))

import HTML.Backend
import HTML.Base

------------------------------------------------------------------------
-- Oracles
------------------------------------------------------------------------

-- | A non-serialisable compile "answer".
data CompileA = CompileA
  { _cState  :: TCState
  , cHash   :: Hash
  }
  deriving Typeable

instance Show CompileA where
  show CompileA{} = "CompileA"

instance Eq CompileA where
  a == b = cHash a == cHash b

instance Hashable CompileA where
  hashWithSalt salt x = hashWithSalt salt (cHash x)

instance Binary CompileA where
  put _ = error "Cannot encode CompileA"
  get = fail "Cannot decode CompileA"

instance NFData CompileA where
  rnf x = cHash x `seq` ()

newtype MainCompileQ = MainCompileQ ()
  deriving (Show, Typeable, Eq, Hashable, Binary, NFData)

type instance RuleResult MainCompileQ = CompileA

newtype ModuleCompileQ = ModuleCompileQ T.Text
  deriving (Show, Typeable, Eq, Hashable, Binary, NFData)

type instance RuleResult ModuleCompileQ = CompileA

------------------------------------------------------------------------
-- Rules and actual compilation
------------------------------------------------------------------------

agdaRules :: Rules ()
agdaRules = do
  -- Add an oracle to compile Agda
  ref <- liftIO $ newIORef Nothing
  _ <- addOracleHash \(MainCompileQ ()) -> compileAgda ref

  -- Add a 'projection' oracle which compiles Agda and then uses the per-module
  -- hash instead. This ensures we only rebuild HTML/types when the module
  -- changes.
  _ <- addOracleHash \(ModuleCompileQ m) -> do
    (CompileA state _) <- askOracle (MainCompileQ ())

    let hash = iFullHash . getInterface state $ toTopLevel state m
    pure (CompileA state hash)

  -- Create a cache for the per-module type information.
  getTypes <- newCache \modName -> do
    let path = "_build/html0" </> modName <.> ".json"
    need [path]

    result :: Either String (Hm.HashMap T.Text Identifier) <-
      quietly . traced "read types" $ eitherDecodeFileStrict' path
    either fail pure result

  -- In order to write the JSON, we first compile the required module (which
  -- gives us the TC environment) and then emit the HTML/Markdown and type JSON.
  "_build/html0/*.json" %> \file -> do
    let modName = T.pack . dropExtension $ takeFileName file
    compileResult <- askOracle (ModuleCompileQ modName)
    emitAgda compileResult getTypes modName

  -- Add a couple of forwarding rules for emitting the actual HTML/MD
  "_build/html0/*.html" %> \file -> need [file -<.> "json"]
  "_build/html0/*.used" %> \file -> need [file -<.> "json"]
  "_build/html0/*.md"   %> \file -> need [file -<.> "json"]

  -- We generate the all-types JSON from the all-pages types JSON - it's just a
  -- schema change.
  "_build/all-types.json" %> \file -> do
    types <- getTypes "all-pages"
    liftIO $ encodeFile file $ map (\t -> t{idTooltip = ""}) (Hm.elems types)

-- | Compile the top-level Agda file.
compileAgda :: IORef (Maybe TCState) -> Action CompileA
compileAgda stateVar = do
  files <- map ("src" </>) <$> getDirectoryFiles "src" ["**/*.agda", "**/*.lagda.md"]
  changed <- needHasChanged $ "_build/all-pages.agda":files

  watching <- getWatching

  traced "agda" do
    baseDir <- absolute "_build"

    -- Reuse the old TC state so we don't reload interfaces each time.
    oldState <- readIORef stateVar

    -- On the initial build, always build everything. Otherwise try to only
    -- rebuild the files which have changed.
    (target, state) <- case oldState of
      Nothing -> (["_build/all-pages.agda"],) <$> initStateIO
      Just state -> pure (if watching then changed else ["_build/all-pages.agda"], state)

    ((), state) <- runTCMPrettyErrors initEnv state do
      -- We preserve the old modules and restore them at the end, as otherwise
      -- we forget them, and will fail if we need to rebuild other HTML pages.
      oldVisited <- useTC stVisitedModules

      resetState

      -- Force Cubical even if we've not got a --cubical header.
      stPragmaOptions `modifyTCLens` \ o -> o { _optCubical = Just CFull }
      setCommandLineOptions' baseDir defaultOptions

      for_ target \source -> do
        absPath <- liftIO $ absolute source
        source <- parseSource =<< srcFromPath absPath
        typeCheckMain TypeCheck source

      modifyTCLens stVisitedModules (`Map.union` oldVisited)

    writeIORef stateVar (Just state)

    -- TODO: Catch errors and pretty-print failures

    pure (CompileA state 0)


-- | Emit an Agda file as HTML or Markdown.
emitAgda
  :: CompileA
  -> (String -> Action (Hm.HashMap T.Text Identifier))
  -> T.Text -> Action ()
emitAgda (CompileA tcState _) getTypes modName = do
  basepn <- filePath <$> liftIO (absolute "src/")

  let
    tlModName = toTopLevel tcState modName
    iface = getInterface tcState tlModName

  types <- parallel . map (getTypes . render . pretty . fst) $ iImportedModules iface

  skipTypes <- getSkipTypes
  ((), _) <- quietly . traced "agda html"
    . runTCM initEnv tcState
    . withScope_ (iInsideScope iface)
    . locallyTC eActiveBackendName (const $ Just "HTML") $ do
      compileOneModule basepn defaultHtmlOptions { htmlOptGenTypes = not skipTypes }
        (mconcat types)
        iface

  pure ()

-- | Convert a module name to an internal Agda name.
--
-- This ia a very cut down version of topLevelModuleName. We sacrifice all the
-- extra checking for the benefit of not having to run a TCM monad.
toTopLevel :: TCState -> T.Text -> TopLevelModuleName
toTopLevel tcState name =
  let
    qname = List1.fromList (T.split (== '.') name)
    raw = RawTopLevelModuleName noRange qname
    hash = BiMap.lookup raw (tcState ^. stTopLevelModuleNames)
    hash' = fromMaybe (hashRawTopLevelModuleName raw) hash
  in
  TopLevelModuleName noRange hash' qname

getInterface :: TCState -> TopLevelModuleName -> Interface
getInterface tcState name =
  let modules = tcState ^. stVisitedModules in
  case Map.lookup name modules of
    Just iface -> miInterface iface
    Nothing -> error $ "Cannot find interface for module " ++ prettyShow name

runTCMPrettyErrors :: TCEnv -> TCState -> TCM a -> IO (a, TCState)
runTCMPrettyErrors env state tcm = do
  (r, state) <- runTCM env state $ (Just <$> tcm) `catchError` \err -> do
    warnings <- fmap (map show) . prettyTCWarnings' =<< getAllWarningsOfTCErr err
    errors  <- show <$> prettyError err
    let everything = filter (not . null) $ warnings ++ [errors]
    unless (null errors) . liftIO . putStr $ unlines everything
    pure Nothing

  case r of
    Just r -> pure (r, state)
    Nothing -> fail "Agda compilation failed"
