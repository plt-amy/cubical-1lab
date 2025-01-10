-- Copyright (c) 2005-2021 remains with the Agda authors. See /support/shake/LICENSE.agda

-- | Backend for generating highlighted, hyperlinked HTML from Agda sources.
{-# LANGUAGE FlexibleContexts, BlockArguments, LambdaCase, DerivingStrategies, OverloadedStrings, NondecreasingIndentation #-}
{-# LANGUAGE DerivingVia #-}
{-# LANGUAGE TypeFamilies #-}
module HTML.Backend
  ( htmlBackend
  , compileOneModule
  , moduleName
  , defaultHtmlOptions
  ) where

import HTML.Base

import Prelude hiding ((!!), concatMap)
import Control.Monad.Identity
import Control.Monad.Except
import Control.Monad.Reader

import qualified Data.HashMap.Strict as Hm
import qualified Data.Map.Strict as Map
import qualified Data.HashSet as HashSet
import qualified Data.Text as Text
import qualified Data.Set as Set

import Data.HashMap.Strict (HashMap)
import Data.List.NonEmpty (NonEmpty((:|)))
import Data.Map.Strict (Map)
import Data.Foldable
import Data.HashSet (HashSet)
import Data.Aeson
import Data.Maybe
import Data.IORef
import Data.Text (Text)
import Data.List
import Data.Set (Set)

import Agda.Compiler.Backend hiding (topLevelModuleName)
import Agda.Compiler.Common

import Control.DeepSeq

import qualified Agda.Syntax.Concrete.Generic as Con
import qualified Agda.Syntax.Internal.Generic as I
import qualified Agda.Syntax.Abstract.Views as A
import qualified Agda.Syntax.Common.Aspect as Asp
import qualified Agda.Syntax.Scope.Base as Scope
import qualified Agda.Syntax.Concrete as Con
import qualified Agda.Syntax.Internal as I
import qualified Agda.Syntax.Abstract as A
import qualified Agda.Syntax.Concrete as C

import Agda.Syntax.Translation.InternalToAbstract ( Reify(reify) )
import Agda.Syntax.Translation.AbstractToConcrete (abstractToConcrete_, abstractToConcreteCtx)
import Agda.Syntax.TopLevelModuleName
import Agda.Syntax.Abstract.Pretty (prettyA, prettyATop)
import Agda.Syntax.Abstract.Views
import Agda.Syntax.Common.Pretty
import Agda.Syntax.Scope.Monad (modifyCurrentScope, getCurrentModule, freshConcreteName)
import Agda.Syntax.Abstract hiding (Type)
import Agda.Syntax.Position
import Agda.Syntax.Internal (Type, domName)
import Agda.Syntax.Fixity (Precedence(TopCtx))
import Agda.Syntax.Common
import Agda.Syntax.Info

import qualified Agda.TypeChecking.Monad.Base as I
import qualified Agda.TypeChecking.Pretty as P
import Agda.TypeChecking.Substitute
import Agda.TypeChecking.Telescope
import Agda.TypeChecking.Records (isRecord)
import Agda.TypeChecking.Reduce (instantiateFull, reduceDefCopyTCM, normalise)
import Agda.TypeChecking.Level (reallyUnLevelView)

import qualified Agda.Utils.Maybe.Strict as S
import Agda.Utils.FileName
import Agda.Utils.Lens
import Agda.Utils.Size

import System.FilePath hiding (normalise)

import Text.Show.Pretty (ppShow)

import HTML.Render
import Debug.Trace (traceShow)
import Agda.Syntax.Internal.Names (namesIn')
import Agda.Interaction.Response (Response_boot(Resp_RunningInfo))
import Agda.Interaction.Options.Lenses (LensVerbosity(setVerbosity))

data HtmlCompileEnv = HtmlCompileEnv
  { htmlCompileEnvOpts     :: HtmlOptions
  , htmlCompileTypes       :: IORef (HashMap Text Identifier)
    -- ^ Hashmap from anchor→identifier for finding types while emitting
    -- HTML, and for search after
  }

data HtmlModuleEnv = HtmlModuleEnv
  { htmlModEnvCompileEnv :: HtmlCompileEnv
  , htmlModEnvName       :: TopLevelModuleName
  }

newtype HtmlModule = HtmlModule { getHtmlModule :: HashMap Text Identifier }

htmlBackend :: FilePath -> HtmlOptions -> Backend
htmlBackend = (Backend .) . htmlBackend'

htmlBackend'
  :: FilePath
  -> HtmlOptions
  -> Backend'
      (FilePath, HtmlOptions)
      HtmlCompileEnv
      HtmlModuleEnv
      HtmlModule
      (Maybe (Text, Identifier))
htmlBackend' basepn opts = Backend'
  { backendName = "HTML"
  , backendVersion = Nothing
  , options = (basepn, opts)
  , commandLineFlags = []
  , isEnabled = const True
  , preCompile = preCompileHtml
  , preModule = preModuleHtml
  , compileDef = compileDefHtml
  , postModule = postModuleHtml
  , postCompile = postCompileHtml
  , scopeCheckingSuffices = False
  , mayEraseType = const $ return False
  , backendInteractTop = Nothing
  , backendInteractHole = Nothing
  }

runLogHtmlWithMonadDebug :: MonadDebug m => LogHtmlT m a -> m a
runLogHtmlWithMonadDebug = runLogHtmlWith $ reportS "html" 1

preCompileHtml
  :: (MonadIO m, MonadDebug m)
  => (FilePath, HtmlOptions)
  -> m HtmlCompileEnv
preCompileHtml (_pn, opts) = do
  types <- liftIO (newIORef mempty)
  runLogHtmlWithMonadDebug $ pure $ HtmlCompileEnv opts types

preModuleHtml
  :: (MonadIO m, ReadTCState m)
  => HtmlCompileEnv
  -> IsMain
  -> TopLevelModuleName
  -> Maybe FilePath
  -> m (Recompile HtmlModuleEnv HtmlModule)
preModuleHtml cenv _isMain topl _ifacePath
  | htmlOptGenTypes (htmlCompileEnvOpts cenv) = do
    liftIO . putStrLn $ "Entering module " <> render (pretty topl)
    pure $ Recompile (HtmlModuleEnv cenv topl)
-- When types are being skipped we can safely only re-render modules
-- whose interface file have changed:
preModuleHtml cenv _ modName mifile =
  do
    ft <- iFileType <$> curIF
    let
      ext = highlightedFileExt (htmlOptHighlight (htmlCompileEnvOpts cenv)) ft
      path = htmlOptDir (htmlCompileEnvOpts cenv) </> modToFile modName ext

    liftIO $ do
      uptd <- uptodate path
      if uptd
        then do
          putStrLn $ "HTML for module " <> render (pretty modName) <> " is up-to-date"
          pure $ Skip (HtmlModule mempty)
        else pure $ Recompile (HtmlModuleEnv cenv modName)

  where
    uptodate of_ = case mifile of
      Nothing -> pure False
      Just ifile -> isNewerThan of_ ifile

compileDefHtml
  :: HtmlCompileEnv
  -> HtmlModuleEnv
  -> IsMain
  -> Definition
  -> TCM (Maybe (Text, Identifier))
compileDefHtml env _ _ _
  | not (htmlOptGenTypes (htmlCompileEnvOpts env)) = pure Nothing
compileDefHtml env _menv _isMain def = withCurrentModule (qnameModule (defName def)) $
  case definitionAnchor env def of
    Just mn -> do
      (tooltip, ty) <- typeToText def
      let
        ident = Identifier
          { idAnchor  = mn
          , idIdent   = Text.pack (render (pretty (qnameName (defName def))))
          , idType    = ty
          , idTooltip = tooltip
          }
      pure (Just (mn, ident))
    Nothing -> do
      pure Nothing

postModuleHtml
  :: (MonadIO m, MonadDebug m, ReadTCState m)
  => HtmlCompileEnv
  -> HtmlModuleEnv
  -> IsMain
  -> TopLevelModuleName
  -> [Maybe (Text, Identifier)]
  -> m HtmlModule
postModuleHtml env menv _isMain _modName _defs = do
  let
    ins Nothing = id
    ins (Just (a, b)) = Hm.insert a b

  types <- liftIO $ atomicModifyIORef' (htmlCompileTypes env) $
    \mp -> let mp' = foldr ins mp _defs in (mp', mp')

  let
    generatePage =
        defaultPageGen types
      . htmlCompileEnvOpts
      . htmlModEnvCompileEnv
      $ menv

  htmlSrc <- srcFileOfInterface (htmlModEnvName menv) <$> curIF
  runLogHtmlWithMonadDebug $ generatePage htmlSrc
  pure $ HtmlModule $ foldr ins mempty _defs

postCompileHtml
  :: MonadIO m
  => HtmlCompileEnv
  -> IsMain
  -> Map TopLevelModuleName HtmlModule
  -> m ()
postCompileHtml cenv _isMain _modulesByName = liftIO $ do
  case htmlOptDumpIdents (htmlCompileEnvOpts cenv) of
    Just fp -> encodeFile fp (Map.elems _modulesByName >>= Hm.elems . getHtmlModule)
    Nothing -> pure ()

-- | Compile a single module, given an existing set of types.
compileOneModule
  :: FilePath -> HtmlOptions
  -> HashMap Text Identifier -- ^ Existing map of identifiers to their types.
  -> Interface -- ^ The interface to compile.
  -> TCM ()
compileOneModule _pn opts types iface = do
  types <- liftIO (newIORef types)
  let cEnv = HtmlCompileEnv opts types
      mEnv = HtmlModuleEnv cEnv (iTopLevelModuleName iface)

  setInterface iface

  defs <- map snd . sortDefs <$> curDefs
  res  <- mapM (compDef cEnv mEnv <=< instantiateFull) defs
  _ <- postModuleHtml cEnv mEnv NotMain (iTopLevelModuleName iface) res
  pure ()

  where
    compDef env menv def = setCurrentRange (defName def) $
      compileDefHtml env menv NotMain def

getClass :: QName -> TCM (Set QName)
getClass q = isRecord q >>= \case
  Just RecordData{ _recFields = f } -> pure $! Set.fromList (map I.unDom f)
  _ -> pure mempty

usedInstances :: I.TermLike a => a -> TCM (Set QName)
usedInstances = I.foldTerm \case
  I.Def q _ -> do
    def <- getConstInfo q
    case Agda.Compiler.Backend.defInstance def of
      Just c  -> Set.insert q <$> getClass (instanceClass c)
      Nothing -> pure mempty
  _ -> pure mempty

typeToText :: Definition -> TCM (Text, Text)
typeToText d = do
  ui <- usedInstances (I.unEl (defType d))
  fv <- getDefFreeVars (defName d)

  ty <- locallyReduceDefs (OnlyReduceDefs ui) $ normalise (defType d)

  topm <- topLevelModuleName (qnameModule (defName d))

  let
    n k = Asp.Name (Just k) False
    asp = \case
      I.Axiom{}          -> n Asp.Postulate
      DataOrRecSig{}     -> n Asp.Datatype
      GeneralizableVar{} -> n Asp.Generalizable
      AbstractDefn d     -> asp d
      d@Function{}
        | isProperProjection d -> n Asp.Field
        | otherwise -> n Asp.Function
      Datatype{}         -> n Asp.Datatype
      Record{}           -> n Asp.Record{}
      Constructor{conSrcCon = c} ->
        n $ Asp.Constructor (I.conInductive c)
      I.Primitive{} -> n Asp.Primitive
      PrimitiveSort{} -> Asp.PrimitiveType

    aspect = asp (theDef d)

    a = Asp.Aspects
      { Asp.aspect         = Just aspect
      , Asp.otherAspects   = mempty
      , Asp.note           = ""
      , Asp.definitionSite = toDefinitionSite topm (nameBindingSite (qnameName (defName d)))
      , Asp.tokenBased     = Asp.TokenBased
      }

  expr <- removeImpls <$> reify ty

  here <- currentTopLevelModule
  tooltip <- fmap renderToHtml $ P.vcat
    [ annotate a <$> P.pretty (qnameName (defName d))
    , P.nest 2 (P.colon P.<+> prettyATop expr)
    ]
  plain <- Text.pack . render <$> prettyATop expr
  pure (tooltip, plain)

toDefinitionSite :: TopLevelModuleName -> Range -> Maybe Asp.DefinitionSite
toDefinitionSite topm r = do
  p <- fmap (fromIntegral . posPos) . rStart $ r
  pure $ Asp.DefinitionSite
    { Asp.defSiteModule = topm
    , Asp.defSitePos    = fromIntegral p
    , Asp.defSiteHere   = True
    , Asp.defSiteAnchor = Nothing
    }

killQual :: Con.Expr -> Con.Expr
killQual = Con.mapExpr wrap where
  work :: Con.QName -> Con.QName
  work (Con.Qual _ x) = work x
  work x = x

  wrap :: Con.Expr -> Con.Expr
  wrap (Con.Ident v)                 = Con.Ident (work v)
  wrap (Con.KnownIdent v w)          = Con.KnownIdent v (work w)
  wrap (Con.KnownOpApp v a b c d)    = Con.KnownOpApp v a (work b) c d
  wrap (Con.OpApp v qual names args) = Con.OpApp v (work qual) names args
  wrap x = x

removeImpls :: A.Expr -> A.Expr
removeImpls (A.Pi _ (x :| xs) e) =
  let
    fixup :: TypedBinding -> TypedBinding
    fixup q@(TBind rng inf as _)
      | Hidden <- getHiding q = TBind rng inf as underscore
      | otherwise = q
  in makePi (map (A.mapExpr removeImpls) $ map fixup (x:xs)) (removeImpls e)
removeImpls (A.Fun span arg ret) =
  A.Fun span (removeImpls <$> arg) (removeImpls ret)
removeImpls e = e

makePi :: [A.TypedBinding] -> A.Expr -> A.Expr
makePi [] e = e
makePi (b:bs) (A.Pi _ bs' e') = makePi (b:bs ++ toList bs') e'
makePi (b:bs) e = A.Pi exprNoRange (b `mergeTBinds` bs) e

mergeTBinds :: A.TypedBinding -> [A.TypedBinding] -> NonEmpty A.TypedBinding
mergeTBinds (A.TBind r i as Underscore{}) (A.TBind _ _ as' Underscore{}:bs) =
  mergeTBinds (A.TBind r i (as <> as') underscore) bs
mergeTBinds x xs = x :| xs

definitionAnchor :: HtmlCompileEnv -> Definition -> Maybe Text
definitionAnchor _ def | defCopy def = Nothing
definitionAnchor _ def = f =<< go where
  go :: Maybe FilePath
  go = do
    let name = defName def
    case rangeModule (nameBindingSite (qnameName name)) of
      Just f -> Just (modToFile f "html")
      Nothing -> Nothing
  f modn =
    case rStart (nameBindingSite (qnameName (defName def))) of
      Just pn -> pure $ Text.pack (modn <> "#" <> show (posPos pn))
      Nothing -> Nothing

-- | Determine the name of a module from a file like @1Lab/HIT/Torus@.
moduleName :: FilePath -> String
moduleName = intercalate "." . splitDirectories
