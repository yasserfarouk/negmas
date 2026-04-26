package genius.core.repository.boa;

import java.io.Serializable;

import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlRootElement;

import genius.core.Global;
import genius.core.boaframework.BOA;
import genius.core.boaframework.BoaType;
import genius.core.exceptions.InstantiateException;
import genius.core.repository.RepItem;

/**
 * Abstract superclass for specific Boa rep items. immutable.
 * 
 * @param T
 *            the type of the contained {@link BOA} component
 */
@XmlRootElement(name = "boa")
@SuppressWarnings("serial")
public class BoaRepItem<T extends BOA> implements Serializable, RepItem {

	/**
	 * Classpath (as file, not a full qualified path) of the item in the
	 * repository
	 */
	@XmlAttribute
	protected String classpath;

	/**
	 * Cached type. Recovered from the actual object
	 */
	private BoaType type = BoaType.UNKNOWN;

	/**
	 * cache of the actual name. null if not yet cached.
	 */
	protected String name = "FAILED TO LOAD";

	private boolean initialized = false;

	// for serializer
	protected BoaRepItem() {
	}

	/**
	 * @param classPath2
	 */
	public BoaRepItem(String classPath2) {
		this.classpath = classPath2;
	}

	@Override
	public String toString() {
		return "boa(" + classpath + ")";
	}

	public BoaType getType() {
		init();
		return type;
	}

	private void init() {
		if (!initialized) {
			initialized = true;
			try {
				BOA instance = getInstance();
				name = instance.getName();
				type = BoaType.typeOf(instance.getClass());
			} catch (InstantiateException e) {
				e.printStackTrace();
			}

		}
	}

	public String getName() {
		init();
		return name;
	}

	/**
	 * @return fully.qualified.class name, or full path (file) to the class
	 *         file.
	 */
	public String getClassPath() {
		return classpath;
	}

	/**
	 * 
	 * @return a new instance of this boa class. Checks that class is of
	 *         expected type.
	 *         {@link BOA#init(genius.core.boaframework.NegotiationSession)} MUST
	 *         be called immediately after this.
	 * @throws InstantiateException
	 *             if class can't be loaded or is not of correct type.
	 */
	@SuppressWarnings("unchecked")
	public T getInstance() throws InstantiateException {
		return (T) Global.loadObject(classpath);
	}

	// public abstract BOA getInstance() throws InstantiateException;

	@Override
	public int hashCode() {
		return 0;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		@SuppressWarnings("rawtypes")
		BoaRepItem other = (BoaRepItem) obj;
		if (classpath == null) {
			if (other.classpath != null)
				return false;
		} else if (!classpath.equals(other.classpath))
			return false;
		if (getName() == null) {
			if (other.getName() != null)
				return false;
		} else if (!getName().equals(other.getName()))
			return false;
		return true;
	}

}
