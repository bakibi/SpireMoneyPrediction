package com.mycompany.app;

import java.util.Vector;

import org.bson.BSONObject;

import com.mongodb.BasicDBObject;
import com.mongodb.DB;
import com.mongodb.DBCollection;
import com.mongodb.DBCursor;
import com.mongodb.DBObject;
import com.mongodb.MongoClient;
import com.mongodb.util.JSON;
public class connexion {

	public static void ConnectToMongoDb()
	{
		 con = new MongoClient("localhost",27017);
		 db=con.getDB("SpireMoneyDB");
		 col = db.getCollection("alphav");
	}
	
	public static void insertJSON(String obj)
	{
		try {
			col.insert((DBObject) JSON.parse(obj));	
		} catch(Exception e) {
			System.out.println("none");
		}
		
	}
	public static Vector<DBObject>  retrieve()
	{
		DBCursor cur = col.find();
		Vector<DBObject>  ans = new Vector<DBObject>(); 
		while(cur.hasNext())
		{
			DBObject o = cur.next();
			System.out.println(o);
			ans.add(o);
		}
		
		return ans;
		
	}
	public static void remove(String data,DBCollection collection)
	{
		BasicDBObject objbasic = new BasicDBObject();
		objbasic.put("open",data );
		col.remove(objbasic);
		
	}
	public static void removeby(String by,String value)
	{
		BasicDBObject document = new BasicDBObject();
		document.put(by, value);
		col.remove(document);
		System.out.println("It has ben removed successfuly\n");
	}
	public static void updateby(String by ,String olddata, String newdata ,DBCollection collection)
	{
		DBObject objbasic = new BasicDBObject();
		objbasic.put(by,olddata );
		DBObject todBobj = new BasicDBObject();
		todBobj.put(by,newdata );
		DBObject updatedBobj = new BasicDBObject();
		updatedBobj.put("$set",todBobj );
		collection.update(objbasic, updatedBobj);
		
		
	}
	private static String teste[]={"2018-02-23","50","67396.15491137","65264.28046911","65940.52223350","141.58517321","0"};
	public static void insertPoint(String record[])
	{  
		
		//collection.insert((DBObject) JSON.parse(data));
		DBObject todBobj = new BasicDBObject();
			todBobj.put("date",record[0]);
			todBobj.put("open",record[1]);
			todBobj.put("high",record[2]);
			todBobj.put("low",record[3]);
			todBobj.put("close",record[4]);
			todBobj.put("volume",record[5]);
			todBobj.put("id",record[6]);
		    
		col.insert(todBobj);
	
	}
	public static void findAll(DBCollection collection)
	{
		DBCursor curs = collection.find();
		while(curs.hasNext())
		{
			System.out.println(curs.next());
		}
	}
	public static void main(String[] args)
	{
		
		connexion.ConnectToMongoDb();
		
		connexion.retrieve();
		
		System.out.println("\n insert ");
		connexion.insertPoint(teste);
		
		
		connexion.updateby("open", "50", "0", col);
		connexion.retrieve();
		
		System.out.println("\n remove ");
		connexion.removeby("open", "10");
		
		connexion.retrieve();
	}
	
	private static MongoClient con;
	private static  DB db;
	private static DBCollection col;
}
