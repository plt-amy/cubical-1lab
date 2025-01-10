{-# LANGUAGE BlockArguments, OverloadedStrings, FlexibleContexts, ScopedTypeVariables #-}
module Main (main) where

import Control.Concurrent.STM
import Control.Monad.IO.Class
import Control.Monad.Writer
import Control.Exception

import qualified Data.Map.Strict as Map
import qualified Data.Set as Set
import Data.Aeson hiding (Options, defaultOptions)
import Data.Bifunctor
import Data.Foldable
import Data.Either
import Data.List

import Development.Shake.FilePath
import Development.Shake.Database
import Development.Shake hiding (getEnv)

import qualified System.Directory as Dir
import qualified System.FSNotify as Watch
import System.Console.GetOpt
import System.Environment
import System.Time.Extra
import System.Process
import System.Exit

import Shake.Options
import Shake.AgdaCompile
import Shake.SearchData
import Shake.LinkGraph
import Shake.Markdown
import Shake.Modules
import Shake.Diagram
import Shake.Digest
import Shake.KaTeX
import Shake.Git
import Shake.Utils

import Definitions
import Timer
import Shake.Recent (recentAdditions)

{-
  Welcome to the Horror That Is 1Lab's Build Script.

  Building 1Lab comprises of (roughly) the following steps:
-}
rules :: Rules ()
rules = do
  agdaRules
  digestRules
  gitRules
  katexRules
  moduleRules
  linksRules
  glossaryRules

  {-
    Write @_build/all-pages.agda@. This imports every module in the source tree
    (causing Agda to compile them) as well as importing all builtin modules
    (which is required for fetching type information).

    Once the file is produced, compile them and write the resulting HTML (for
    @.agda@) and markdown (for @.lagda.md@) files to @_build/html0@.
  -}
  "_build/all-pages.agda" %> \out -> do
    modules <- Map.toList <$> getOurModules
    let
      toOut (x, WithText) = x ++ " -- (text page)"
      toOut (x, CodeOnly) = x ++ " -- (code only)"

    writeFileLines out $ ["open import " ++ toOut x | x <- modules]

  {-
    For each 1Lab module, read the emitted file from @_build/html0@. If it's
    HTML, we just copy it to @_build/html@. Otherwise we compile the markdown
    to HTML with some additional post-processing steps (see 'buildMarkdown')
  -}
  "_build/html/*.html" %> \out -> do
    let modName = dropExtension (takeFileName out)

    modKind <- Map.lookup modName <$> getOurModules
    skipAgda <- getSkipAgda
    let
      input
        | not skipAgda = "_build/html0" </> modName
        | otherwise = case modName of
          "all-pages" -> "_build/all-pages"
          _ -> "src" </> map (\c -> if c == '.' then '/' else c) modName

      inext = case modKind of
        Just WithText
          | skipAgda  -> "lagda.md"
          | otherwise -> "md"
        _ -> if skipAgda then "agda" else "html"

    case modKind of
      Just WithText -> buildMarkdown modName (input <.> inext) out
      _ -> copyFile' (input <.> inext) out -- Wrong, but eh!

    unless skipAgda $ need ["_build/html/types" </> modName <.> "json"]

  "_build/search/*.json" %> \out -> need ["_build/html" </> takeFileName out -<.> "html"]
  "_build/html/types/*.json" %> \out -> do
    let
      mn = takeFileName out
      it = "_build/html0/" </> mn -<.> "used"
    need [it]
    copyFile' it $ "_build/html/types/" </> mn

  "_build/html/static/search.json" %> \out -> do
    skipAgda <- getSkipAgda
    modules <- filter ((==) WithText . snd) . Map.toList <$> getOurModules
    let searchFiles = (if skipAgda then [] else ["_build/all-types.json"])
                    ++ map (\(x, _) -> "_build/search" </> x <.> "json") modules
    searchData :: [[SearchTerm]] <- traverse readJSONFile searchFiles
    traced "Writing search data" $ encodeFile out (concat searchData)

  -- Compile Quiver to SVG. This is used by 'buildMarkdown'.
  "_build/html/**/*.light.svg" %> \out -> do
    let
      inp = "_build/diagrams"
        </> takeFileName (takeDirectory out)
        </> takeBaseName out -<.> "tex"
    buildDiagram (getPreambleFor False) inp out False

  "_build/html/**/*.dark.svg" %> \out -> do
    let
      inp = "_build/diagrams"
        </> takeFileName (takeDirectory out)
        </> takeBaseName out -<.> "tex"
    buildDiagram (getPreambleFor True) inp out True

  "_build/html/css/*.css" %> \out -> do
    let inp = "support/web/css/" </> takeFileName out -<.> "scss"
    getDirectoryFiles "support/web/css" ["**/*.scss"] >>= \files -> need ["support/web/css" </> f | f <- files]
    command_ [] "sass" [inp, out]

  "_build/html/favicon.ico" %> \out -> do
    need ["support/favicon.ico"]
    copyFile' "support/favicon.ico" out

  "_build/html/static/**/*" %> \out -> do
    let inp = "support/" </> dropDirectory1 (dropDirectory1 out)
    need [inp]
    traced "copying" $ Dir.copyFile inp out

  "_build/html/*.js" %> \out -> do
    getDirectoryFiles "support/web/js" ["**/*.ts", "**/*.tsx"] >>= \files -> need ["support/web/js" </> f | f <- files]

    let inp = "support/web/js" </> takeFileName out -<.> "ts"
    nodeCommand [] "esbuild"
      [ "--bundle", inp
      , "--outfile=" ++ out
      , "--target=es2017"
      , "--minify"
      , "--sourcemap"
      ]

  phony "preamble" do
    liftIO . print =<< getPreambleFor True
    liftIO . print =<< getParsedPreamble

  {-
    The final build step. This basically just finds all the files we actually
    need and kicks off the above job to build them.
  -}
  phony "all" do
    skipAgda <- getSkipAgda
    agda <- getAllModules >>= \modules -> pure do
      (f, _) <- Map.toList modules
      [ "_build/html" </> f <.> "html" ] <>
        [ "_build/html/types" </> f <.> "json" | not skipAgda ]
    static <- getDirectoryFiles "support/static/" ["**/*"] >>= \files ->
      pure ["_build/html/static" </> f | f <- files]
    need $
      static ++ agda ++
        [ "_build/html/favicon.ico"
        , "_build/html/static/links.json"
        , "_build/html/static/search.json"
        , "_build/html/css/default.css"
        , "_build/html/css/talk.css"
        , "_build/html/main.js"
        , "_build/html/start.js"
        , "_build/html/code-only.js"
        ]

  -- ???

  phony "clean" do
    removeFilesAfter "_build" ["html0/*", "html/*", "diagrams/*", "*.agda", "*.md", "*.html"]

  phony "really-clean" do
    need ["clean"]
    removeFilesAfter "_build" ["**/*.agdai", "*.lua"]

  phony "typecheck-ts" do
    getDirectoryFiles "support/web/js" ["**/*.ts", "**/*.tsx"] >>= \files -> need ["support/web/js" </> f | f <- files]
    nodeCommand [] "tsc" ["--noEmit", "-p", "tsconfig.json"]

  phony "recent" do liftIO . print =<< recentAdditions

  -- Profit!

main :: IO ()
main = do
  args <- getArgs
  when ("--help" `elem` args || "-h" `elem` args) do
    putStrLn $ usageInfo "shake" optDescrs
    exitFailure

  let (opts, extra, errs) = getOpt Permute optDescrs args
      (shakeOpts, ourOpts_) = partitionEithers opts
      (errs', shakeOpts') = first (++errs) $ partitionEithers shakeOpts
      ourOpts = foldr (.) id ourOpts_ defaultOptions

      rules' = do
        setOptions ourOpts
        rules

      shakeOptions' = foldl' (flip ($)) shakeOptions{shakeFiles="_build", shakeChange=ChangeDigest} shakeOpts'
      (shakeRules, wanted) = case extra of
        [] -> (rules', [])
        files -> (withoutActions rules', [need files])

  unless (null errs') do
    for_ errs' $ putStrLn . ("1lab-shake: " ++)
    exitFailure

  (ok, after) <- shakeWithDatabase shakeOptions' shakeRules \db -> do
    case _optWatching ourOpts of
      Nothing  -> buildOnce db wanted
      Just cmd -> buildMany db wanted cmd
  shakeRunAfter shakeOptions' after

  reportTimes

  unless ok exitFailure

  where
    optDescrs :: [OptDescr (Either (Either String (ShakeOptions -> ShakeOptions)) (Options -> Options))]
    optDescrs = map (fmap Left) shakeOptDescrs ++ map (fmap Right) _1LabOptDescrs

    buildOnce :: ShakeDatabase -> [Action ()] -> IO (Bool, [IO ()])
    buildOnce db wanted = do
      start <- offsetTime

      res :: Either SomeException x <- try $ shakeRunDatabase db wanted

      tot <- start
      case res of
        Left err -> do
          putStrLn $ "❌ Build failed in " ++ showDuration tot
          print err
          pure (False, [])
        Right (_, actions) -> do
          putStrLn $ "✅ Build succeeded in " ++ showDuration tot
          pure (True, actions)

    -- | Watch config with 10ms (0.01 / 1e-12)
    watchConfig :: Watch.WatchConfig
    watchConfig = Watch.defaultConfig

    isFile Watch.IsFile      = True
    isFile Watch.IsDirectory = False

    buildMany :: ShakeDatabase -> [Action ()] -> Maybe String -> IO (Bool, [IO ()])
    buildMany db wanted cmd = Watch.withManagerConf watchConfig \mgr -> do
      (_, clean) <- buildOnce db wanted

      toRebuild <- atomically $ newTVar Set.empty
      _ <- Watch.watchTree mgr "src" (isFile . Watch.eventIsDirectory) (logEvent toRebuild)
      _ <- Watch.watchTree mgr "support" (isFile . Watch.eventIsDirectory) (logEvent toRebuild)

      root <- Dir.canonicalizePath "."

      let
        loop clean = do
          changes <- atomically do
            changes <- swapTVar toRebuild Set.empty
            when (Set.null changes) retry
            pure changes

          -- Some editors write temporary, non-module files in this dir while saving, which triggers
          -- a full rebuild. Prune our set of changes to files which still exist.
          changes' <- map (makeRelative root) <$> filterM Dir.doesFileExist (Set.toList changes)

          if null changes' then loop clean else do
            let
              -- If all our changed files are Agda modules, try to emit just those
              -- HTML files, rather than everything.
              (targets, targetName) = case traverse toModule changes' of
                Nothing -> (wanted, "everything")
                Just [] -> (wanted, "everything")
                Just xs ->
                  let targets = map (\x -> "_build/html" </> x <.> "html") xs
                  in ([need targets], intercalate ", " targets)

            putStrLn $ "🔨 " ++ intercalate ", " changes' ++ " has changed. Rebuilding " ++ targetName ++ "."

            (_, clean') <- buildOnce db targets

            case cmd of
              Nothing -> pure ()
              Just cmd -> do
                putStrLn $ "🔨 Running `" ++ cmd ++ "`"
                _ <- withCreateProcess (shell cmd) { delegate_ctlc = True } (\_ _ _ p -> waitForProcess p)
                pure ()

            loop (clean' ++ clean)

      loop clean

    logEvent toRebuild event = atomically $ modifyTVar' toRebuild (Set.insert (Watch.eventPath event))

    toModule :: FilePath -> Maybe String
    toModule path
      | (path, ext) <- splitExtensions path
      , ext == ".lagda.md" || ext == ".agda"
      , ("src":rest) <- splitDirectories path
      = Just $ intercalate "." rest
      | otherwise = Nothing
