package genius.core;

import java.util.Calendar;
import java.util.Date;
import java.util.HashMap;
import java.util.TimeZone;
import java.util.regex.Matcher;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.Serializable;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLClassLoader;
import java.text.SimpleDateFormat;

import javax.swing.JOptionPane;

import genius.core.exceptions.InstantiateException;
import genius.core.protocol.Protocol;
import genius.core.repository.AgentRepItem;
import genius.core.repository.ProfileRepItem;
import genius.core.repository.ProtocolRepItem;
import genius.core.tournament.TournamentConfiguration;
import genius.core.tournament.VariablesAndValues.AgentParamValue;
import genius.core.tournament.VariablesAndValues.AgentParameterVariable;
import genius.gui.agentrepository.AgentRepositoryUI;

/**
 * Overview of global variables used throughout the application.
 * 
 * @author dmytro
 */
public class Global {
	/** Path to domain repository */
	public static final String DOMAIN_REPOSITORY = "domainrepository.xml";
	/** Path to agent repository */
	public static final String AGENT_REPOSITORY = "agentrepository.xml";
	/** Path to protocol repository */
	public static final String PROTOCOL_REPOSITORY = "protocolrepository.xml";
	/** Path to simulator repository */
	public static final String SIMULATOR_REPOSITORY = "simulatorrepository.xml";

	public static String logPrefix = "";

	public static String logPreset = "";

	private final static String WRONG_NAME = "wrong name: ";

	private static final Date loadDate = Calendar.getInstance().getTime();

	public static String getCurrentTime() {
		Calendar cal = Calendar.getInstance(TimeZone.getDefault());
		String DATE_FORMAT = "yyyy-MM-dd HH:mm:ss";
		java.text.SimpleDateFormat sdf = new java.text.SimpleDateFormat(DATE_FORMAT);
		/*
		 * on some JDK, the default TimeZone is wrong we must set the TimeZone
		 * manually.
		 */
		sdf.setTimeZone(TimeZone.getDefault());

		return sdf.format(cal.getTime());
	}

	public static String getFileNameWithoutExtension(String fileName) {

		File tmpFile = new File(fileName);
		tmpFile.getName();
		int whereDot = tmpFile.getName().lastIndexOf('.');
		if (0 < whereDot && whereDot <= tmpFile.getName().length() - 2) {
			return tmpFile.getName().substring(0, whereDot);
		}
		return "";
	}

	public static Class<Protocol> getProtocolClass(ProtocolRepItem protRepItem) throws Exception {
		java.lang.ClassLoader loader = Global.class.getClassLoader();// ClassLoader.getSystemClassLoader();
		Class<Protocol> klass = (Class<Protocol>) loader.loadClass(protRepItem.getClassPath());
		return klass;
	}

	public static Protocol createProtocolInstance(ProtocolRepItem protRepItem, AgentRepItem[] agentRepItems,
			ProfileRepItem[] profileRepItems, HashMap<AgentParameterVariable, AgentParamValue>[] agentParams)
			throws InstantiateException {
		try {
			Protocol ns;

			java.lang.ClassLoader loader = ClassLoader.getSystemClassLoader();

			Class klass;
			klass = loader.loadClass(protRepItem.getClassPath());
			Class[] paramTypes = { AgentRepItem[].class, ProfileRepItem[].class, HashMap[].class, int.class };

			Constructor cons = klass.getConstructor(paramTypes);

			System.out.println("Found the constructor: " + cons);

			Object[] args = { agentRepItems, profileRepItems, agentParams, 1 };

			Object theObject = cons.newInstance(args);
			ns = (Protocol) (theObject);
			return ns;
		} catch (ClassNotFoundException | NoSuchMethodException | SecurityException | InstantiationException
				| IllegalAccessException | IllegalArgumentException | InvocationTargetException e) {
			throw new InstantiateException("Failed to create instance", e);
		}
	}

	/**
	 * Load an object from a given path. If it's a .class file, figure out the
	 * correct class path and use that. If it's not a .class file, we assume
	 * it's already in the existing classpath and load it with the standard
	 * class loader.
	 * 
	 * 
	 * <p>
	 * we can't properly typecheck here. Generics fail as we have type erasure,
	 * and casting to the given type does NOTHING. So we leave this a general
	 * object and leave it to the caller to do the type checking.
	 * 
	 * @param path
	 *            This can be either a class name or filename.<br>
	 *            <ul>
	 *            <li>class name like"agents.anac.y2010.AgentFSEGA.AgentFSEGA".
	 *            In this case the agent must be already on the JVM's classpath
	 *            otherwise the agent will not be found. <br>
	 *            <li>a full path, eg
	 *            "/Volumes/documents/NegoWorkspace3/NegotiatorGUI/src/agents/anac/y2010/AgentFSEGA/AgentFSEGA.java"
	 *            . In this case, we can figure out the class path ourselves,
	 *            but the ref is system dependent (backslashes on windows) and
	 *            might be absolute path.
	 *            </ul>
	 * 
	 * @return the {@link Object} in the given file
	 * @throws InstantiateException
	 *             if path can not be loaded as object.
	 */
	public static Object loadObject(String path) throws InstantiateException {
		try {
			if (path.endsWith(".class")) {
				return loadClassFromFile(new File(path));
			} else {
				java.lang.ClassLoader loaderA = Global.class.getClassLoader();
				return (loaderA.loadClass(path).newInstance());
			}
		} catch (Exception e) {
			throw new InstantiateException("failed to load class from  " + path, e);
		}
	}

	/**
	 * Runtime type-checked version of {@link #loadObject(String)}.
	 * 
	 * @param path
	 * @param expectedClass
	 *            the class type that the loaded object must extend.
	 * @return loaded object.
	 * @throws InstantiateException
	 */

	public static Object loadObject(String path, Class<?> expectedClass) throws InstantiateException {
		Object object = loadObject(path);
		if (!object.getClass().isAssignableFrom(expectedClass)) {
			throw new InstantiateException("Failed to load class " + path + ": It is not extending " + expectedClass);
		}
		return object;
	}

	/**
	 * Deserializes an object and casts it to the given type.
	 * 
	 * @param is
	 *            the input stream containing serialized object.
	 * @return object contained in given stream.
	 * @throws IOException
	 *             if file can not be found
	 * @throws ClassNotFoundException
	 *             if class in the object can't be found
	 * @throws ClassCastException
	 *             if not of given class type
	 */
	@SuppressWarnings("unchecked")
	public static <T> T deserializeObject(InputStream is) throws ClassNotFoundException, IOException {
		Object obj = new ObjectInputStream(is).readObject();
		return (T) obj;
	}

	/**
	 * Serialize a serializable object to a outputstream.
	 * 
	 * @param outputStream
	 *            the stream to write to
	 * @param object
	 *            the object to store
	 * @throws IOException
	 */
	public static void serializeObject(OutputStream outputStream, Serializable object) throws IOException {
		new ObjectOutputStream(outputStream).writeObject(object);
	}

	/**
	 * Load a file as a class. It 'reverse engineers' the correct path by first
	 * just trying to load the file. Assuming the file exists, we probably get
	 * an error that we then use to determine the correct base directory.
	 * 
	 * @param file
	 *            the object to be loaded. Filename should end with ".class".
	 * @return the object contained in the file.
	 * @throws ClassNotFoundException
	 * @throws IllegalAccessException
	 * @throws InstantiationException
	 * @throws MalformedURLException
	 */
	public static Object loadClassFromFile(File file)
			throws MalformedURLException, InstantiationException, IllegalAccessException, ClassNotFoundException {
		String className = file.getName();
		if (!className.endsWith(".class")) {
			throw new IllegalArgumentException("file " + file + " is not a .class file");
		}
		// strip the trailing '.class' from the string.
		className = className.substring(0, className.length() - 6);
		File packageDir = file.getParentFile();
		if (packageDir == null) {
			packageDir = new File(".");
		}

		try {
			// System.out.println("Loading " + className + " from " + packageDir);
			// return loadClassfile("bilateralexamples.BoaPartyExample", "T:\\workspace\\Genius10test\\bin");
			return loadClassfile(className, packageDir);
		} catch (NoClassDefFoundError e) {
			/**
			 * We try to get the correct name from the error message. Err msg ~
			 * "SimpleAgent (wrong name: agents/SimpleAgent)"
			 */
			String errormsg = e.getMessage();
			// "wrong name" is what we expect.
			int i = errormsg.indexOf(WRONG_NAME);
			if (i == -1) {
				throw e; // unknown error. We can't handle...
			}
			// remove leading and trailing stuff. We now have
			// 'agents.SimpleAgent'
			String correctName = errormsg.substring(i + WRONG_NAME.length(), errormsg.length() - 1).replaceAll("/",
					".");

			// Check that file is in correct directory path
			// we need quoteReplacement because on Windows "\" will be treated
			// in special way by replaceAll. #906
			String expectedPath = File.separator
					+ correctName.replaceAll("\\.", Matcher.quoteReplacement(File.separator)) + ".class";
			if (!(file.getAbsolutePath().endsWith(expectedPath))) {
				throw new NoClassDefFoundError("file " + file + "\nis not in the correct directory structure, "
						+ "\nas its class is " + correctName + "." + "\nEnsure the file is in ..." + expectedPath);
			}

			// number of dots is number of times we need to go to parent
			// directory. We are already in the directory of the agent, so -1.
			for (int up = 0; up < correctName.split("\\.").length - 1; up++) {
				// since we checked the path already, parents must exist.
				packageDir = packageDir.getParentFile();
			}
			try 
			{
				// System.out.println("Tried fixing error. Now loading " + correctName + " from " + packageDir);
				return loadClassfile(correctName, packageDir);
			}
			catch (NoClassDefFoundError f) 
			{
				correctName = className;
				
				while (true)
				{
					try 
					{
						/*
						 * We now try fixing:
						 * Loading BoaPartyExample from T:\workspace\Genius10test\bin\bilateralexamples
						 * 
						 * changing that to:
						 * Loading bilateralexamples.BoaPartyExample from T:\workspace\Genius10test\bin
						 */
						File parentDir = packageDir.getParentFile(); 	// T:\workspace\Genius10test\bin
						String packageName = packageDir.getName(); 		// bilateralexamples
						correctName = packageName + "." + correctName; 	// bilateralexamples.BoaPartyExample
						packageDir = parentDir;

						if (packageDir == null)
							throw new IllegalArgumentException("Could not find any class file " + correctName + " in parents of " + parentDir);

//						System.out.println("Tried final way of fixing error by examining the parent directory. Now loading " + correctName + " from " + packageDir);
						return loadClassfile(correctName, packageDir);
					}
					catch (NoClassDefFoundError g) 
					{
						// continue
					}
				}
			}
		}
	}


	/**
	 * Try to load an object with given classnamem from a given packagedir
	 * 
	 * @param classname
	 *            the exact class name, eg "examplepackage.example"
	 * @param packagedir
	 *            the root directory of the classes to be loaded. If you add the
	 *            given classname to it, you should end up at the correct
	 *            location for the class file. Eg,
	 *            "/Volumes/Users/wouter/Desktop/genius/".
	 * @return the loaded class object.
	 * @throws MalformedURLException
	 * @throws InstantiationException
	 * @throws IllegalAccessException
	 * @throws ClassNotFoundException
	 */
	private static Object loadClassfile(String classname, File packagedir)
			throws MalformedURLException, InstantiationException, IllegalAccessException, ClassNotFoundException {
		try {
			java.lang.ClassLoader loader = AgentRepositoryUI.class.getClassLoader();
			URLClassLoader urlLoader = new URLClassLoader(new URL[] { packagedir.toURI().toURL() }, loader);
			Class<?> theclass;
			theclass = urlLoader.loadClass(classname);
			return (Object) theclass.newInstance();
		} catch (ClassNotFoundException e) {
			// improve on the standard error message...
			throw new ClassNotFoundException(
					"Agent " + classname + " is not available in directory '" + packagedir + "'", e);
		}

	}

	/**
	 * Load an agent using the given classname/filename. DOES NOT call
	 * {@link Agent#parseStrategyParameters(String)}
	 * 
	 * @param path
	 *            This can be either a class name or filename.<br>
	 *            <ul>
	 *            <li>class name like"agents.anac.y2010.AgentFSEGA.AgentFSEGA".
	 *            In this case the agent must be already on the JVM's classpath
	 *            otherwise the agent will not be found.
	 *            <li>a full path, eg
	 *            "/Volumes/documents/NegoWorkspace3/NegotiatorGUI/src/agents/anac/y2010/AgentFSEGA/AgentFSEGA.java"
	 *            . In this case, we can figure out the class path ourselves,
	 *            but the ref is system dependent (backslashes on windows) and
	 *            might be absolute path.
	 *            </ul>
	 * @return instantiated agent ready to use.
	 * @throws InstantiateException
	 *             if object can't be loaded
	 */
	public static Agent loadAgent(String path) throws InstantiateException {
		return (Agent) loadObject(path);
	}

	/**
	 * load agent and then set the parameters. See {@link #loadAgent(String)}
	 * 
	 * @param agentClassName
	 * @param variables
	 *            the variables to use, as string (eg, "time=0.9;e=1.0").
	 * @return the agent contained in the given class name, and using the given
	 *         variables.
	 * @throws InstantiateException
	 *             if class can't be loaded
	 */
	public static Agent loadAgent(String agentClassName, String variables) throws InstantiateException {

		Agent agent = loadAgent(agentClassName);

		// CHECK why do we catch failures in parseStrategyParameters?
		try {
			agent.parseStrategyParameters(variables);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return agent;

	}

	/**
	 * Gives a useful agent name.
	 */
	public static String getAgentDescription(Agent agent) {
		if (agent == null)
			return "";
		String agentDescription = agent.getName();
		if (agentDescription == null || "Agent A".equals(agentDescription) || "Agent B".equals(agentDescription))
			agentDescription = agent.getClass().getSimpleName();

		return agentDescription;
	}

	/**
	 * Show a dialog to the user, explaining the exception that was raised while
	 * loading file fc. Typically this is used in combination with
	 * {@link #loadObject(String)} and associates. Also dumps a copy of the full
	 * stacktrace to the console, to help us debugging #906
	 * 
	 * @param fc
	 *            file that was attempted to be loaded
	 * @param e
	 *            the exception that was raised
	 */
	public static void showLoadError(File fc, Throwable e) {
		e.printStackTrace();
		if (e instanceof ClassNotFoundException) {
			showLoadError("No class found at " + fc, e);
		} else if (e instanceof InstantiationException) {
			// happens when object instantiated is interface or abstract
			showLoadError(
					"Class cannot be instantiated. Reasons may be that there is no constructor without arguments, "
							+ "or the class is abstract or an interface.",
					e);
		} else if (e instanceof IllegalAccessException) {
			showLoadError("Missing constructor without arguments", e);
		} else if (e instanceof NoClassDefFoundError) {
			showLoadError("Errors in loaded class.", e);
		} else if (e instanceof ClassCastException) {
			showLoadError("The loaded class seems to be of the wrong type. ", e);
		} else if (e instanceof IllegalArgumentException) {
			showLoadError("The given file can not be used.", e);
		} else if (e instanceof IOException) {
			showLoadError("The file can not be read.", e);
		} else {
			showLoadError("Something went wrong loading the file", e);
		}
	}

	/*
	 * show error while loading agent file. Also show the detail message.
	 */
	private static void showLoadError(String text, Throwable e) {
		String message = e.getMessage();
		if (message == null) {
			message = "";
		}

		JOptionPane.showMessageDialog(null, text + "\n" + message, "Load error", 0);
	}

	/**
	 * @return the agentsLoader
	 */
	private static String getLoadDate() {
		// (2) createFrom our "formatter" (our custom format)
		SimpleDateFormat formatter = new SimpleDateFormat("yyyy-MM-dd HH.mm.ss");

		// (3) createFrom a new String in the format we want
		String name = formatter.format(loadDate);

		return name;
	}

	public static String getOutcomesFileName() {
		if (!logPreset.equals("")) {
			return logPreset;
		}
		if (!logPrefix.equals(""))
			return logPrefix + "log.xml";

		return "log/" + getLoadDate() + getPostFix() + ".xml";
	}

	public static String getDistributedOutcomesFileName() {
		return "log/DT-" + getLoadDate() + getPostFix() + ".xml";
	}

	public static String getTournamentOutcomeFileName() {
		return "log/TM-" + getLoadDate() + getPostFix() + ".xml";
	}

	public static String getExtensiveOutcomesFileName() {
		if (!logPrefix.equals(""))
			return logPrefix + "extensive_log.xml";
		return "log/extensive " + getLoadDate() + getPostFix() + ".xml";
	}

	public static String getOQMOutcomesFileName() {
		return "log/OQM " + getLoadDate() + getPostFix() + ".csv";
	}

	private static String getPostFix() {
		String postFix = "";
		if (TournamentConfiguration.getBooleanOption("appendModeAndDeadline", false)) {
			String mode = "time";
			if (TournamentConfiguration.getBooleanOption("protocolMode", false)) {
				mode = "rounds";
			}
			postFix += "_" + mode + "_" + TournamentConfiguration.getIntegerOption("deadline", 60);
		}
		return postFix;
	}

	/**
	 * @param classname
	 * @return Removes trailing ".class" from string if it is there. In absolute
	 *         paths, the \ and / are replaced with '.' and we do as if that is
	 *         a fully specified class path (it isn't but it gives at least some
	 *         'short name')
	 */
	public static String nameOfClass(String classname1) {
		// FIXME can we use class.forName.getShortName?
		String classname = classname1.replaceAll("\\W", ".");
		if (classname.endsWith(".class")) {
			classname = classname.substring(0, classname.length() - 6);
		}
		return classname;
	}

	/**
	 * @param classname
	 * @return Removes trailing ".class" from string if it is there. If there is
	 *         no "." in the remainder, the remainder is returned. Otherwise,
	 *         the string after the last "." in the remainder is returned. In
	 *         absolute paths, the \ and / are replaced with '.' and we do as if
	 *         that is a fully specified class path (it isn't but it gives at
	 *         least some 'short name')
	 */
	public static String shortNameOfClass(String classname1) {
		String classname = nameOfClass(classname1);
		if (!(classname.contains(".")))
			return classname;
		return classname.substring(classname.lastIndexOf(".") + 1);
	}

}
