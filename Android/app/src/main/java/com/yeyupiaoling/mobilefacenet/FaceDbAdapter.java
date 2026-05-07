package com.yeyupiaoling.mobilefacenet;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import java.util.List;

public class FaceDbAdapter extends RecyclerView.Adapter<FaceDbAdapter.ViewHolder> {

    private List<FaceRecognizer.FaceDbEntry> entries;
    private OnItemClickListener listener;

    public interface OnItemClickListener {
        void onEditClick(FaceRecognizer.FaceDbEntry entry, int position);
        void onDeleteClick(FaceRecognizer.FaceDbEntry entry, int position);
    }

    public FaceDbAdapter(List<FaceRecognizer.FaceDbEntry> entries, OnItemClickListener listener) {
        this.entries = entries;
        this.listener = listener;
    }

    public void setEntries(List<FaceRecognizer.FaceDbEntry> entries) {
        this.entries = entries;
        notifyDataSetChanged();
    }

    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.item_face_db, parent, false);
        return new ViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
        FaceRecognizer.FaceDbEntry entry = entries.get(position);
        holder.tvName.setText(entry.name);
        
        // 主判断：确保文件存在后再加载图片
        if (entry.file != null && entry.file.exists()) {
            Bitmap bitmap = BitmapFactory.decodeFile(entry.file.getAbsolutePath());
            holder.ivFace.setImageBitmap(bitmap);
        } else {
            holder.ivFace.setImageDrawable(null);
        }

        holder.btnEdit.setOnClickListener(v -> {
            if (listener != null) {
                listener.onEditClick(entry, position);
            }
        });

        holder.btnDelete.setOnClickListener(v -> {
            if (listener != null) {
                listener.onDeleteClick(entry, position);
            }
        });
    }

    @Override
    public int getItemCount() {
        return entries == null ? 0 : entries.size();
    }

    public static class ViewHolder extends RecyclerView.ViewHolder {
        ImageView ivFace;
        TextView tvName;
        Button btnEdit;
        Button btnDelete;

        public ViewHolder(@NonNull View itemView) {
            super(itemView);
            ivFace = itemView.findViewById(R.id.ivFace);
            tvName = itemView.findViewById(R.id.tvName);
            btnEdit = itemView.findViewById(R.id.btnEdit);
            btnDelete = itemView.findViewById(R.id.btnDelete);
        }
    }
}