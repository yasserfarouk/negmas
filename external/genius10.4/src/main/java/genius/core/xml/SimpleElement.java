package genius.core.xml;
/*
 * @(#)SimpleElement.java
 */

import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.ListIterator;
import java.util.Map;
import java.util.Vector;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Serializable;

/**
 * <code>SimpleElement</code> is the only node type for
 * simplified DOM model.
 */
public class SimpleElement implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = -575978416803618689L;
	protected String tagName;
	protected String text;
	protected HashMap<String, String> attributes;
	protected LinkedList<SimpleElement> childElements;

	public SimpleElement(String tagName) {
		this.tagName = tagName;
		attributes = new HashMap<String, String>();
		childElements = new LinkedList<SimpleElement>();
	}

	public String getTagName() {
		return tagName;
	}

	public void setTagName(String tagName) {
		this.tagName = tagName;
	}

	public String getText() {
		return text;
	}

	public void setText(String text) {
		this.text = text;
	}

	public String getAttribute(String name) {
		return attributes.get(name);
	}

	public HashMap<String, String> getAttributes() {
		return attributes;
	}
	
	public void setAttribute(String name, String value) {
		attributes.put(name, value);
	}

	public void addChildElement(SimpleElement element) {
		childElements.add(element);
	}

	public Object[] getChildElements() {
		return childElements.toArray();
	}
	
	public LinkedList<SimpleElement> getChildElementsAsList() {
		return childElements;
	}
	
	public boolean isEmpty()
	{
		return attributes.isEmpty() && childElements.isEmpty();
	}
	
	public void combineLists(HashMap<String,String> element){
		//System.out.println("Testing");
		//setAttribute("Key", "String");
		attributes.put("Testing", "Here");
		Iterator it = element.entrySet().iterator();
	    while (it.hasNext()) {
	        Map.Entry pairs = (Map.Entry)it.next();
			setAttribute(pairs.getKey().toString(), pairs.getValue().toString());

	        
	        //System.out.println(pairs.getKey() + " = " + pairs.getValue());
	        it.remove(); // avoids a ConcurrentModificationException
	    }

		//HashMap<String, String> temp = attributes;
		//attributes.putAll(temp);
		//attributes.putAll(element);
	}
        
	public Object[] getChildByTagName(String tagName) {
	//	LinkedList<Object> result = new LinkedList<Object>();
		Vector<Object> result = new Vector<Object>();
        ListIterator<SimpleElement> iter = childElements.listIterator();
        while(iter.hasNext()) {
        	SimpleElement se = iter.next();
           	String seTagName = se.getTagName();
            if (seTagName.equals(tagName))
				result.add(se);
        }
		Object[] resultArray = new Object[result.size()];//for some reason the toArray gave me a direct reference to the last element of the returned array, not the array itself. - Hdv.
		for(int ind=0; ind < result.size(); ind++){
			resultArray[ind] = result.elementAt(ind);	
		}
		return resultArray;
    }
        
	public String toString() 
	{
		StringBuffer s;
		if (childElements.isEmpty())
			s = new StringBuffer(64);
		else
			s = new StringBuffer(1024);
			
        	s.append("<");
        	s.append(tagName);
            //save all attributes
            for(int i=0;i<attributes.size();i++) {
            	String lAttrName = (String)(attributes.keySet().toArray()[i]);
            	String lAttrValue="";
            	if (attributes.entrySet().toArray()[i]!=null)
            		lAttrValue= (attributes.get(lAttrName));
            	
            	s.append(" "); 
            	s.append(lAttrName);
            	s.append("=\"");
            	s.append(lAttrValue);
            	s.append("\"");
            }
            s.append("> \n");
            //save all children
            for(int i=0;i<childElements.size();i++) {
            	SimpleElement lSE = (SimpleElement)getChildElements()[i];
            	s.append(lSE.toString());
            }
            if(text!=null) {
            	s.append(text);
            	s.append(" \n");
            	}
            s.append("</");
            s.append(tagName);
            s.append("> \n");
        	
        	return s.toString();
        }
        public void saveToFile(String pFileName) {
        	try {
                BufferedWriter out = new BufferedWriter(new FileWriter(pFileName));
                String lXML = toString();
                out.write(lXML);
                out.close();
            } catch (IOException e) {
            	e.printStackTrace();
            }
        	
        }
}
